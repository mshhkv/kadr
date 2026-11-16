//! A 3D scene drawn by the core (AE.md, M6): triangles with a depth buffer, lit as a physically based material, one
//! shadow-casting spot light, an environment, and the frame built from several samples — each with its own moment
//! inside the shutter, its own point on the lens (depth of field), its own sub-pixel shift (anti-aliasing) and its own
//! place of the lamp (soft shadows). Python (kadr/scene3d.py) decides what is where at each sample; this file only
//! draws. The shaders are in wgsl/scene3d*.wgsl.
use std::collections::HashMap;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::{gpu, img, Gpu, Img};

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const GSIZE: u64 = 1152;                    // struct G of scene3d.wgsl
const GSTRIDE: u64 = 1280;                  // …at the offset alignment of a uniform (256)
const ISIZE: u64 = 144;                     // struct Inst
const MSIZE: u64 = 80;                      // struct Mat
const SHADOW: u32 = 2048;

struct Targets {
    w: u32,
    h: u32,
    sample: wgpu::TextureView,
    depth: wgpu::TextureView,
    acc: wgpu::TextureView,
    levels: Vec<(wgpu::TextureView, u32, u32)>,
}

#[pyclass(unsendable)]
pub struct Scene3d {
    g: &'static Gpu,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    geos: Vec<(u32, u32, i32)>,             // first index, how many, base vertex
    maps: Vec<wgpu::BindGroup>,             // 0: a white pixel
    l0: wgpu::BindGroupLayout,
    ls: wgpu::BindGroupLayout,
    lpost: wgpu::BindGroupLayout,
    lfinal: wgpu::BindGroupLayout,
    pl: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    pipes: HashMap<u32, wgpu::RenderPipeline>,
    shadow_pipes: [wgpu::RenderPipeline; 2],
    add_pipe: wgpu::RenderPipeline,
    down_pipe: wgpu::RenderPipeline,
    final_pipe: wgpu::ComputePipeline,
    shadow_view: wgpu::TextureView,
    cmp: wgpu::Sampler,
    lin: wgpu::Sampler,
    env: wgpu::TextureView,
    ltc: [wgpu::TextureView; 2],
    dfg: wgpu::TextureView,
    targets: Option<Targets>,
}

fn tex(g: &Gpu, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, mips: u32) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn write(g: &Gpu, t: &wgpu::Texture, level: u32, w: u32, h: u32, bytes: &[u8], bpp: u32) {
    g.queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: t, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * bpp), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

/// a picture of half floats (h, w, 4) → a texture that can be read with linear filtering
fn half_tex(g: &Gpu, a: &PyReadonlyArray3<u16>) -> PyResult<wgpu::TextureView> {
    let s = a.shape();
    if s[2] != 4 {
        return Err(PyRuntimeError::new_err("нужен массив (h, w, 4) uint16"));
    }
    let t = tex(g, s[1] as u32, s[0] as u32, HDR, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 1);
    let data = a.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    write(g, &t, 0, s[1] as u32, s[0] as u32, bytemuck::cast_slice(data), 8);
    Ok(t.create_view(&Default::default()))
}

fn entry(binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None }
}

fn tex_ty(filterable: bool) -> wgpu::BindingType {
    wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }
}

fn uniform_ty(size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: std::num::NonZeroU64::new(size) }
}

fn storage_ty(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }
}

const VERTEX: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
];

fn vlayout<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout { array_stride: 32, step_mode: wgpu::VertexStepMode::Vertex, attributes: &VERTEX }
}

impl Scene3d {
    /// a pipeline of the main pass. key: bit 0 — both sides; bit 1 — no depth writing; bits 2–3 — blending (0 none,
    /// 1 alpha, 2 additive); bit 4 — a sprite
    fn pipe(&mut self, key: u32) -> wgpu::RenderPipeline {
        if let Some(p) = self.pipes.get(&key) {
            return p.clone();
        }
        let blend = match (key >> 2) & 3 {
            1 => Some(wgpu::BlendState {
                color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            }),
            2 => Some(wgpu::BlendState {
                color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
                alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            }),
            _ => None,
        };
        let sprite = key & 16 != 0;
        let p = self.g.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene3d"),
            layout: Some(&self.pl),
            vertex: wgpu::VertexState { module: &self.module, entry_point: Some(if sprite { "vs_sprite" } else { "vs" }), compilation_options: Default::default(), buffers: &[Some(vlayout())] },
            primitive: wgpu::PrimitiveState { cull_mode: if key & 1 != 0 || sprite { None } else { Some(wgpu::Face::Back) }, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(key & 2 == 0),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &self.module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: HDR, blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.pipes.insert(key, p.clone());
        p
    }

    fn targets(&mut self, w: u32, h: u32) {
        if matches!(&self.targets, Some(t) if t.w == w && t.h == h) {
            return;
        }
        let g = self.g;
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let view = |t: wgpu::Texture| t.create_view(&Default::default());
        let mut levels = Vec::new();
        let (mut lw, mut lh) = (w, h);
        for _ in 0..6 {
            lw = (lw >> 1).max(2);
            lh = (lh >> 1).max(2);
            levels.push((view(tex(g, lw, lh, HDR, ra, 1)), lw, lh));
        }
        self.targets = Some(Targets { w, h, sample: view(tex(g, w, h, HDR, ra, 1)), depth: view(tex(g, w, h, DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT, 1)),
                                      acc: view(tex(g, w, h, HDR, ra, 1)), levels });
    }
}

fn ub(b: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: b, offset: 0, size: std::num::NonZeroU64::new(size) })
}

fn tv(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(v)
}

fn post_bg(d: &wgpu::Device, layout: &wgpu::BindGroupLayout, pbuf: &wgpu::Buffer, v: &wgpu::TextureView, lin: &wgpu::Sampler) -> wgpu::BindGroup {
    d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: ub(pbuf, 64) },
        wgpu::BindGroupEntry { binding: 1, resource: tv(v) },
        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(lin) }] })
}

fn post_pipe(g: &Gpu, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str, add: bool) -> wgpu::RenderPipeline {
    let one = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
    g.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("scene3d post"),
        layout: Some(layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format: HDR, blend: if add { Some(wgpu::BlendState { color: one, alpha: one }) } else { None }, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

#[pymethods]
impl Scene3d {
    /// verts (n, 8): position, normal, uv; indices; geos: (first index, count, base vertex) of each model;
    /// maps: the materials' pictures — each a list of mip levels (h, w, 4) uint8, sRGB or not, repeating or not;
    /// sets: for each material the numbers of its four pictures among `maps` (−1: none) — colour, roughness and metal
    /// (green and blue; red: how much surrounding light reaches it), relief (a normal map), glow. A draw names its set
    /// by number + 1 (0: no pictures). Empty: one set a picture, as its colour;
    /// env, ltc_1, ltc_2, dfg: half-float pictures (h, w, 4) — the prefiltered environment (CubeUV), the two tables of
    /// the rect-area light, the table of a rough mirror's reflectance
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(verts: PyReadonlyArray2<f32>, indices: PyReadonlyArray1<u32>, geos: Vec<(u32, u32, i32)>, maps: Vec<(Vec<PyReadonlyArray3<u8>>, bool, bool)>, sets: Vec<[i32; 4]>,
           env: PyReadonlyArray3<u16>, ltc_1: PyReadonlyArray3<u16>, ltc_2: PyReadonlyArray3<u16>, dfg: PyReadonlyArray3<u16>) -> PyResult<Scene3d> {
        let g = gpu()?;
        let d = &g.device;
        let buf = |data: &[u8], usage| {
            let b = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: (data.len() as u64).max(16).next_multiple_of(4), usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            g.queue.write_buffer(&b, 0, data);
            b
        };
        let v = verts.as_slice().map_err(|_| PyRuntimeError::new_err("verts: массив должен быть непрерывным"))?;
        let ix = indices.as_slice().map_err(|_| PyRuntimeError::new_err("indices: массив должен быть непрерывным"))?;
        let vbuf = buf(bytemuck::cast_slice(v), wgpu::BufferUsages::VERTEX);
        let ibuf = buf(bytemuck::cast_slice(ix), wgpu::BufferUsages::INDEX);
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let f = wgpu::ShaderStages::FRAGMENT;
        let l0 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d"),
            entries: &[entry(0, vf, uniform_ty(GSIZE)), entry(1, vf, storage_ty(true)), entry(2, f, storage_ty(true)),
                       entry(3, f, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                       entry(4, f, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)),
                       entry(5, f, tex_ty(true)), entry(6, f, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                       entry(7, f, tex_ty(true)), entry(8, f, tex_ty(true)), entry(9, f, tex_ty(true))],
        });
        let l1 = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d map"),
            entries: &[entry(0, f, tex_ty(true)), entry(1, f, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                       entry(2, f, tex_ty(true)), entry(3, f, tex_ty(true)), entry(4, f, tex_ty(true))],
        });
        let ls = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d shadow"),
            entries: &[entry(0, wgpu::ShaderStages::VERTEX, uniform_ty(GSIZE)), entry(1, wgpu::ShaderStages::VERTEX, storage_ty(true))],
        });
        let lpost = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d post"),
            entries: &[entry(0, f, uniform_ty(64)), entry(1, f, tex_ty(true)), entry(2, f, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering))],
        });
        let c = wgpu::ShaderStages::COMPUTE;
        let lfinal = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene3d final"),
            entries: &[entry(0, c, uniform_ty(64)), entry(1, c, tex_ty(true)), entry(2, c, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                       entry(3, c, tex_ty(true)), entry(4, c, tex_ty(true)), entry(5, c, tex_ty(true)), entry(6, c, tex_ty(true)), entry(7, c, tex_ty(true)), entry(8, c, storage_ty(false))],
        });
        let layout = |ls: &[Option<&wgpu::BindGroupLayout>]| d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("scene3d"), bind_group_layouts: ls, immediate_size: 0 });
        let pl = layout(&[Some(&l0), Some(&l1)]);
        let pls = layout(&[Some(&ls)]);
        let plpost = layout(&[Some(&lpost)]);
        let plfinal = layout(&[Some(&lfinal)]);
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scene3d"), source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/scene3d.wgsl").into()) });
        // the shadow pass binds only what a vertex needs: the same structures, cut down
        let shadow_src = "struct G { vp: mat4x4<f32>, view: mat4x4<f32>, proj: mat4x4<f32>, shadow_vp: mat4x4<f32>, rest: array<vec4<f32>, 56> };\n\
            struct Inst { model: mat4x4<f32>, nrm: mat4x4<f32>, m: vec4<u32> };\n\
            @group(0) @binding(0) var<uniform> g: G;\n@group(0) @binding(1) var<storage, read> insts: array<Inst>;\n\
            @vertex fn vs(@location(0) pos: vec3<f32>, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {\n\
                return g.shadow_vp * (insts[ii].model * vec4<f32>(pos, 1.0));\n}\n";
        let smodule = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scene3d shadow"), source: wgpu::ShaderSource::Wgsl(shadow_src.into()) });
        let post = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scene3d post"), source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/scene3d_post.wgsl").into()) });
        let fmodule = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("scene3d final"), source: wgpu::ShaderSource::Wgsl(include_str!("wgsl/scene3d_final.wgsl").into()) });
        let spipe = |cull: Option<wgpu::Face>| d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene3d shadow"),
            layout: Some(&pls),
            vertex: wgpu::VertexState { module: &smodule, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[Some(vlayout())] },
            primitive: wgpu::PrimitiveState { cull_mode: cull, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH, depth_write_enabled: Some(true), depth_compare: Some(wgpu::CompareFunction::LessEqual), stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        // a one-sided thing goes into the shadow map by its back faces (as three.js does it: less self-shadowing)
        let shadow_pipes = [spipe(Some(wgpu::Face::Front)), spipe(None)];
        let add_pipe = post_pipe(g, &plpost, &post, "fs_add", true);
        let down_pipe = post_pipe(g, &plpost, &post, "fs_down", false);
        let final_pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scene3d final"), layout: Some(&plfinal), module: &fmodule, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(PyRuntimeError::new_err(format!("scene3d: шейдер не собрался: {e}")));
        }
        let clampy = wgpu::AddressMode::ClampToEdge;
        let lin = d.create_sampler(&wgpu::SamplerDescriptor { address_mode_u: clampy, address_mode_v: clampy, address_mode_w: clampy,
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let cmp = d.create_sampler(&wgpu::SamplerDescriptor { address_mode_u: clampy, address_mode_v: clampy, address_mode_w: clampy,
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, compare: Some(wgpu::CompareFunction::LessEqual), ..Default::default() });
        let shadow_view = tex(g, SHADOW, SHADOW, DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, 1).create_view(&Default::default());
        // the materials' pictures: views[0] a white pixel (no picture: the number it multiplies stays), views[1] a flat
        // relief, then the scene's own
        let mut views: Vec<wgpu::TextureView> = Vec::new();
        let mut smps: Vec<wgpu::Sampler> = Vec::new();
        let mut add_map = |levels: Vec<(u32, u32, Vec<u8>)>, srgb: bool, repeat: bool| {
            let fmt = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
            let t = tex(g, levels[0].0, levels[0].1, fmt, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, levels.len() as u32);
            for (k, (w, h, bytes)) in levels.iter().enumerate() {
                write(g, &t, k as u32, *w, *h, bytes, 4);
            }
            let am = if repeat { wgpu::AddressMode::Repeat } else { clampy };
            smps.push(d.create_sampler(&wgpu::SamplerDescriptor { address_mode_u: am, address_mode_v: am, address_mode_w: am, mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::MipmapFilterMode::Linear, anisotropy_clamp: 16, ..Default::default() }));
            views.push(t.create_view(&Default::default()));
        };
        add_map(vec![(1, 1, vec![255u8; 4])], false, false);
        add_map(vec![(1, 1, vec![128u8, 128, 255, 255])], false, false);
        for (levels, srgb, repeat) in &maps {
            let mut ls = Vec::new();
            for a in levels {
                let s = a.shape();
                ls.push((s[1] as u32, s[0] as u32, a.as_slice().map_err(|_| PyRuntimeError::new_err("maps: массив должен быть непрерывным"))?.to_vec()));
            }
            add_map(ls, *srgb, *repeat);
        }
        let n_maps = maps.len() as i32;
        let sets: Vec<[i32; 4]> = if sets.is_empty() { (0..n_maps).map(|i| [i, -1, -1, -1]).collect() } else { sets };
        let pick = |i: i32, none: usize| if i >= 0 && i < n_maps { i as usize + 2 } else { none };
        let mut groups = Vec::new();
        let mut group = |s: [i32; 4]| {
            let (a, b, c, e) = (pick(s[0], 0), pick(s[1], 0), pick(s[2], 1), pick(s[3], 0));
            let smp = [a, b, c, e].into_iter().find(|&k| k >= 2).unwrap_or(0);       // how its pictures repeat: as the first it has
            groups.push(d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &l1, entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&views[a]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&smps[smp]) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&views[b]) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&views[c]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&views[e]) }] }));
        };
        group([-1, -1, -1, -1]);
        for s in &sets {
            group(*s);
        }
        Ok(Scene3d { g, vbuf, ibuf, geos, maps: groups, l0, ls, lpost, lfinal, pl, module, pipes: HashMap::new(), shadow_pipes, add_pipe, down_pipe, final_pipe,
                     shadow_view, cmp, lin, env: half_tex(g, &env)?, ltc: [half_tex(g, &ltc_1)?, half_tex(g, &ltc_2)?], dfg: half_tex(g, &dfg)?, targets: None })
    }

    /// one frame. globals (samples, 288): struct G of each sample; insts (rows, 36): struct Inst; mats (rows, 20): struct Mat;
    /// draws (rows, 6): sample, pass (0 shadow, 1 main), pipeline key, model, row of insts, map — in drawing order;
    /// look: w, h, exposure, glow, tint r g b, filmic gain, vignette from, to, depth, grain, grain seed, glow threshold, raw (1: linear light as it is)
    fn render(&mut self, w: u32, h: u32, globals: PyReadonlyArray2<f32>, insts: PyReadonlyArray2<f32>, mats: PyReadonlyArray2<f32>,
              draws: PyReadonlyArray2<u32>, look: [f32; 15], clear: [f32; 4]) -> PyResult<Img> {
        let g = self.g;
        let d = &g.device;
        let n = globals.shape()[0];
        if globals.shape()[1] * 4 != GSIZE as usize || insts.shape()[1] * 4 != ISIZE as usize || mats.shape()[1] * 4 != MSIZE as usize || draws.shape()[1] != 6 {
            return Err(PyRuntimeError::new_err("scene3d: не тот размер строк globals / insts / mats / draws"));
        }
        self.targets(w, h);
        g.flush();
        let gl = globals.as_slice().map_err(|_| PyRuntimeError::new_err("globals: массив должен быть непрерывным"))?;
        let mut gbytes = vec![0u8; (n as u64 * GSTRIDE) as usize];
        for s in 0..n {
            let src: &[u8] = bytemuck::cast_slice(&gl[s * 288..(s + 1) * 288]);
            gbytes[s * GSTRIDE as usize..s * GSTRIDE as usize + GSIZE as usize].copy_from_slice(src);
        }
        let mk = |data: &[u8], usage| {
            let b = d.create_buffer(&wgpu::BufferDescriptor { label: None, size: (data.len() as u64).max(256).next_multiple_of(4), usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            g.queue.write_buffer(&b, 0, data);
            b
        };
        let gbuf = mk(&gbytes, wgpu::BufferUsages::UNIFORM);
        let ibuf = mk(bytemuck::cast_slice(insts.as_slice().map_err(|_| PyRuntimeError::new_err("insts: массив должен быть непрерывным"))?), wgpu::BufferUsages::STORAGE);
        let mbuf = mk(bytemuck::cast_slice(mats.as_slice().map_err(|_| PyRuntimeError::new_err("mats: массив должен быть непрерывным"))?), wgpu::BufferUsages::STORAGE);
        // post: one small block per pass — n adds, 6 halvings, the final
        let mut pbytes = vec![0u8; (n + 7) * 256];
        let mut put = |k: usize, v: [f32; 16]| pbytes[k * 256..k * 256 + 64].copy_from_slice(bytemuck::cast_slice(&v));
        for s in 0..n {
            let mut v = [0f32; 16];
            v[0] = 1.0 / n as f32;
            put(s, v);
        }
        let t = self.targets.as_ref().unwrap();
        for (l, (_, lw, lh)) in t.levels.iter().enumerate() {
            let mut v = [0f32; 16];
            v[0] = 0.5 / *lw as f32; v[1] = 0.5 / *lh as f32; v[2] = if l == 0 { look[13] } else { 0.0 };
            put(n + l, v);
        }
        put(n + 6, [w as f32, h as f32, look[2], look[3], look[4], look[5], look[6], look[7], look[8], look[9], look[10], look[11], look[12], look[14], 0.0, 0.0]);
        let pbuf = mk(&pbytes, wgpu::BufferUsages::UNIFORM);
        let bg0 = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.l0, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ub(&gbuf, GSIZE) },
            wgpu::BindGroupEntry { binding: 1, resource: ibuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: mbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tv(&self.shadow_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.cmp) },
            wgpu::BindGroupEntry { binding: 5, resource: tv(&self.env) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&self.lin) },
            wgpu::BindGroupEntry { binding: 7, resource: tv(&self.ltc[0]) },
            wgpu::BindGroupEntry { binding: 8, resource: tv(&self.ltc[1]) },
            wgpu::BindGroupEntry { binding: 9, resource: tv(&self.dfg) }] });
        let bgs = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.ls, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ub(&gbuf, GSIZE) },
            wgpu::BindGroupEntry { binding: 1, resource: ibuf.as_entire_binding() }] });
        let dr = draws.as_slice().map_err(|_| PyRuntimeError::new_err("draws: массив должен быть непрерывным"))?;
        // the pipelines the frame needs, made before the passes borrow them
        let mut keys: Vec<u32> = dr.chunks(6).filter(|r| r[1] == 1).map(|r| r[2]).collect();
        keys.sort_unstable();
        keys.dedup();
        let pipes: HashMap<u32, wgpu::RenderPipeline> = keys.into_iter().map(|k| (k, self.pipe(k))).collect();
        let t = self.targets.as_ref().unwrap();
        let sample_bg = post_bg(d, &self.lpost, &pbuf, &t.sample, &self.lin);
        let mut enc = d.create_command_encoder(&Default::default());
        let color = |view, load| Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store } });
        let depth = |view| Some(wgpu::RenderPassDepthStencilAttachment { view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None });
        let zero = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
        for s in 0..n {
            let off = [(s as u64 * GSTRIDE) as u32];
            let mine = |pass: u32| dr.chunks(6).filter(move |r| r[0] == s as u32 && r[1] == pass);
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("shadow"), color_attachments: &[], depth_stencil_attachment: depth(&self.shadow_view),
                    timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_bind_group(0, &bgs, &off);
                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for r in mine(0) {
                    let (first, count, base) = self.geos[r[3] as usize];
                    rp.set_pipeline(&self.shadow_pipes[(r[2] & 1) as usize]);
                    rp.draw_indexed(first..first + count, base, r[4]..r[4] + 1);
                }
            }
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("scene"),
                    color_attachments: &[color(&t.sample, wgpu::LoadOp::Clear(wgpu::Color { r: clear[0] as f64, g: clear[1] as f64, b: clear[2] as f64, a: clear[3] as f64 }))],
                    depth_stencil_attachment: depth(&t.depth), timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_bind_group(0, &bg0, &off);
                rp.set_vertex_buffer(0, self.vbuf.slice(..));
                rp.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for r in mine(1) {
                    let (first, count, base) = self.geos[r[3] as usize];
                    rp.set_pipeline(&pipes[&r[2]]);
                    rp.set_bind_group(1, &self.maps[(r[5] as usize).min(self.maps.len() - 1)], &[]);
                    rp.draw_indexed(first..first + count, base, r[4]..r[4] + 1);
                }
            }
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("add"), color_attachments: &[color(&t.acc, if s == 0 { zero } else { wgpu::LoadOp::Load })],
                    depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_pipeline(&self.add_pipe);
                rp.set_bind_group(0, &sample_bg, &[(s * 256) as u32]);
                rp.draw(0..3, 0..1);
            }
        }
        // the glow: the frame halved six times, the first halving keeping only what is brighter than the threshold
        let mut src = post_bg(d, &self.lpost, &pbuf, &t.acc, &self.lin);
        for (l, (view, _, _)) in t.levels.iter().enumerate() {
            {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("down"), color_attachments: &[color(view, zero)], depth_stencil_attachment: None,
                    timestamp_writes: None, occlusion_query_set: None, multiview_mask: None });
                rp.set_pipeline(&self.down_pipe);
                rp.set_bind_group(0, &src, &[((n + l) * 256) as u32]);
                rp.draw(0..3, 0..1);
            }
            src = post_bg(d, &self.lpost, &pbuf, view, &self.lin);
        }
        let out = img(g, w, h);
        let mut fe = vec![wgpu::BindGroupEntry { binding: 0, resource: ub(&pbuf, 64) }, wgpu::BindGroupEntry { binding: 1, resource: tv(&t.acc) },
                          wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.lin) }];
        for k in 0..5 {
            fe.push(wgpu::BindGroupEntry { binding: 3 + k as u32, resource: tv(&t.levels[k + 1].0) });
        }
        fe.push(wgpu::BindGroupEntry { binding: 8, resource: out.buf.as_entire_binding() });
        let bgf = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.lfinal, entries: &fe });
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.final_pipe);
            cp.set_bind_group(0, &bgf, &[((n + 6) * 256) as u32]);
            cp.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        g.queue.submit([enc.finish()]);
        Ok(out)
    }
}
