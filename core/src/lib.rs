//! Kadr's render core. Python (kadr/render.py) still decides what each frame shows and how — placements, opacity,
//! blur, shadows, masks, groups, motion-blur samples; this crate does the pixel work on the GPU (wgpu: Metal on macOS)
//! with the same formulas, so a frame stays on the GPU from its first layer to the 8-bit picture handed to the encoder.
//!
//! Images live in GPU buffers as `vec4<f32>` per pixel — a layer's own picture as (rgb 0..255 not premultiplied,
//! alpha 0..1), a canvas as (colour accumulated premultiplied, alpha) — the very pairs the Python renderer uses, so
//! every kernel is a line-by-line port of a numpy expression there.
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;

mod text;
mod scene3d;

const COMMON: &str = include_str!("wgsl/common.wgsl");
const KERNELS: &[(&str, &str)] = &[
    ("composite", include_str!("wgsl/composite.wgsl")),
    ("gauss", include_str!("wgsl/gauss.wgsl")),
    ("pm", include_str!("wgsl/pm.wgsl")),
    ("blit", include_str!("wgsl/blit.wgsl")),
    ("lanczos_h", include_str!("wgsl/lanczos_h.wgsl")),
    ("lanczos_v", include_str!("wgsl/lanczos_v.wgsl")),
    ("dither", include_str!("wgsl/dither.wgsl")),
    ("grade", include_str!("wgsl/grade.wgsl")),
    ("matte", include_str!("wgsl/matte.wgsl")),
    ("from8", include_str!("wgsl/from8.wgsl")),
    ("fill", include_str!("wgsl/fill.wgsl")),
    ("project", concat!(include_str!("wgsl/project_fns.wgsl"), include_str!("wgsl/project.wgsl"))),
    ("project_mb", concat!(include_str!("wgsl/project_fns.wgsl"), include_str!("wgsl/project_mb.wgsl"))),
    ("colorize", include_str!("wgsl/colorize.wgsl")),
    ("gradient", include_str!("wgsl/gradient.wgsl")),
    ("boxdown", include_str!("wgsl/boxdown.wgsl")),
    ("cubic", include_str!("wgsl/cubic.wgsl")),
    ("tile", include_str!("wgsl/tile.wgsl")),
    ("prep8", include_str!("wgsl/prep8.wgsl")),
    ("lzw", include_str!("wgsl/lzw.wgsl")),
    ("to8", include_str!("wgsl/to8.wgsl")),
    ("straight8", include_str!("wgsl/straight8.wgsl")),
    ("mip", include_str!("wgsl/mip.wgsl")),
    ("spread", include_str!("wgsl/spread.wgsl")),
    ("gdir", include_str!("wgsl/gdir.wgsl")),
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    f: [f32; 32],
    u: [u32; 16],
}

impl Params {
    fn new() -> Self {
        Params { f: [0.0; 32], u: [0; 16] }
    }
}

/// one step of a frame's GPU work, recorded and sent to the GPU in a batch
enum Cmd {
    Run { kernel: &'static str, p: Params, src: Option<wgpu::Buffer>, dst: wgpu::Buffer, aux: Option<wgpu::Buffer>,
          src8: Option<wgpu::Buffer>, groups: (u32, u32), hs: bool, hd: bool,
          pipe: Option<wgpu::ComputePipeline> },     // a shader from outside the crate (Gpu::dyn_pipe), else the kernel's own
    Clear(wgpu::Buffer),
    Copy(wgpu::Buffer, wgpu::Buffer, u64),
}

const PSTRIDE: u64 = 256;                   // a uniform block per step, at the offset alignment Metal asks for
// steps recorded before they go to the GPU (so it starts while Python works). Small, and at most one command buffer
// waiting behind the running one: a command buffer runs to its end before the window server gets the GPU, and 96 steps
// were ~75 ms of a 15 s reel — the screen and the player stalled up to 180 ms behind 3 render processes. 8 steps,
// 1 in flight (15.11.2026, 375 frames, 3 processes): 91 → 78 s, the worst stall of a compositor-like client 183 → 29 ms
const BATCH: usize = 8;
const INFLIGHT: usize = 1;

/// steps per command buffer (KADR_GPU_BATCH, default BATCH) and command buffers allowed in flight before the next one
/// waits (KADR_GPU_INFLIGHT, default INFLIGHT, 0 = no limit)
fn knob(name: &str, dflt: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(dflt)
}
static KNOBS: OnceLock<(usize, usize)> = OnceLock::new();
fn knobs() -> (usize, usize) {
    *KNOBS.get_or_init(|| (knob("KADR_GPU_BATCH", BATCH).max(1), knob("KADR_GPU_INFLIGHT", INFLIGHT)))
}
const POOL_BYTES: u64 = 768 << 20;          // pictures freed and kept for reuse, at most

/// freed picture buffers by size class: a new picture takes one instead of a fresh allocation (which wgpu also clears)
#[derive(Default)]
struct Pool {
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    bytes: u64,
    resting: Vec<wgpu::Buffer>,             // freed since the last submit: steps not yet sent may still read them, and an
                                            // upload (queue.write_buffer) lands before the next submit's steps — so they
                                            // are reused only after that submit
}

/// size class: up to 1/8 of the power of two above, so a few sizes cover the many picture sizes of a film
fn size_class(bytes: u64) -> u64 {
    let b = bytes.max(256);
    let step = (b.next_power_of_two() / 8).max(256);
    b.div_ceil(step) * step
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    modules: HashMap<&'static str, wgpu::ShaderModule>,
    pl: wgpu::PipelineLayout,
    pipes: Mutex<HashMap<(&'static str, bool, bool), wgpu::ComputePipeline>>,   // per kernel and HS / HD, made when first used
    dpipes: Mutex<HashMap<(String, bool, bool), wgpu::ComputePipeline>>,        // shaders from outside the crate, by key
    dummy: wgpu::Buffer,
    dummy_w: wgpu::Buffer,                  // the writable binding not used (dst or dst16)
    half: Mutex<std::collections::HashSet<u64>>,   // the buffers that now hold a half-precision picture
    name: String,
    pending: Mutex<Vec<Cmd>>,
    pool: Mutex<Pool>,
    timing: bool,                           // KADR_GPU_TIMING: every step in its own pass between two timestamps
    staging: Mutex<Vec<wgpu::Buffer>>,     // read-back buffers, kept for the next frames
    inflight: Mutex<std::collections::VecDeque<wgpu::SubmissionIndex>>,   // submitted, maybe not done (INFLIGHT)
}

/// per kernel: steps and threads dispatched since the last `stats()` — where the GPU's work goes
static STATS: Mutex<Option<HashMap<&'static str, (u64, u64, u64)>>> = Mutex::new(None);
thread_local! {
    // a device (and its queue, pool and pipelines) per thread: the render threads of one process each get a queue of
    // their own, as separate processes would — the GPU runs their small kernels side by side, where one queue runs them
    // one after another (16.11.2026: 2 threads on one device ≈ 1 process; on two devices ≈ 3 processes). Leaked: a
    // thread's device lives as long as the process
    static GPU: std::cell::OnceCell<Result<&'static Gpu, String>> = const { std::cell::OnceCell::new() };
}

fn gpu() -> PyResult<&'static Gpu> {
    GPU.with(|c| c.get_or_init(|| Gpu::new().map(|g| &*Box::leak(Box::new(g)))).clone()).map_err(PyRuntimeError::new_err)
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Gpu {
    fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("нет видеокарты: {e}"))?;
        let name = adapter.get_info().name;
        let limits = adapter.limits();                  // the M4 allows buffers far beyond WebGPU's defaults
        let ts = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("kadr"),
            required_features: ts,
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("видеокарта не открылась: {e}"))?;
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kadr"),
            entries: &[uniform, storage(1, true), storage(2, false), storage(3, true), storage(4, true), storage(5, true), storage(6, false)],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kadr"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut modules = HashMap::new();
        for (name, body) in KERNELS {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{body}").into()),
            });
            modules.insert(*name, module);
        }
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dummy_w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_w"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let timing = !ts.is_empty() && std::env::var_os("KADR_GPU_TIMING").is_some();
        Ok(Gpu { device, queue, layout, modules, pl, pipes: Mutex::new(HashMap::new()), dpipes: Mutex::new(HashMap::new()), dummy, dummy_w, half: Mutex::new(Default::default()), name, pending: Mutex::new(Vec::new()), pool: Mutex::new(Pool::default()), timing, staging: Mutex::new(Vec::new()), inflight: Mutex::new(Default::default()) })
    }

    /// the kernel's pipeline for pictures read (HS) / written (HD) in half precision
    fn pipe(&self, kernel: &'static str, hs: bool, hd: bool) -> wgpu::ComputePipeline {
        let mut ps = self.pipes.lock().unwrap();
        ps.entry((kernel, hs, hd)).or_insert_with(|| {
            let c = [("HS", hs as u32 as f64), ("HD", hd as u32 as f64)];
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel),
                layout: Some(&self.pl),
                module: &self.modules[kernel],
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &c, ..Default::default() },
                cache: None,
            })
        }).clone()
    }

    /// a kernel written outside the crate (kadr/shader.py): `code` goes after common.wgsl and is compiled once per
    /// key and format; a mistake in it comes back as text (the WGSL compiler's message), not as a panic
    fn dyn_pipe(&self, key: &str, code: &str, hs: bool, hd: bool) -> Result<wgpu::ComputePipeline, String> {
        let k = (key.to_string(), hs, hd);
        if let Some(p) = self.dpipes.lock().unwrap().get(&k) {
            return Ok(p.clone());
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{code}").into()),
        });
        let c = [("HS", hs as u32 as f64), ("HD", hd as u32 as f64)];
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shader"),
            layout: Some(&self.pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &c, ..Default::default() },
            cache: None,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(e.to_string());
        }
        self.dpipes.lock().unwrap().insert(k, pipe.clone());
        Ok(pipe)
    }

    fn is_half(&self, b: &wgpu::Buffer) -> bool {
        self.half.lock().unwrap().contains(&bkey(b))
    }

    /// a picture buffer: a freed one of the same size class if there is one, else a new one. Its old contents stay —
    /// every kernel writes its whole output, and a picture that must start empty is cleared on purpose
    fn buffer(&self, bytes: u64) -> wgpu::Buffer {
        let c = size_class(bytes);
        let mut pool = self.pool.lock().unwrap();
        if let Some(b) = pool.free.get_mut(&c).and_then(|v| v.pop()) {
            pool.bytes -= c;
            drop(pool);
            self.half.lock().unwrap().remove(&bkey(&b));     // f32 unless its new owner says half (img_h)
            return b;
        }
        drop(pool);
        let b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: c,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.half.lock().unwrap().remove(&bkey(&b));         // a fresh buffer may reuse a dropped one's identity
        b
    }

    /// a freed picture's buffer back to the pool (steps already recorded keep reading it in order — the GPU runs them
    /// one after another, so a later step that reuses it sees the right thing)
    fn give_back(&self, buf: wgpu::Buffer) {
        let c = buf.size();
        if c != size_class(c) {
            return;                             // an upload's exact-size buffer: not a size class, not pooled
        }
        let mut pool = self.pool.lock().unwrap();
        if pool.bytes + c > POOL_BYTES {
            return;
        }
        pool.bytes += c;
        pool.resting.push(buf);
    }

    /// after a submit: what was freed before it can take new contents now
    fn wake(&self) {
        let mut pool = self.pool.lock().unwrap();
        let rest = std::mem::take(&mut pool.resting);
        for b in rest {
            pool.free.entry(b.size()).or_default().push(b);
        }
    }

    /// bytes → a pooled buffer, through the queue's staging (one copy, no fresh mapped allocation per upload)
    fn upload(&self, data: &[u8]) -> wgpu::Buffer {
        let buf = self.buffer(data.len() as u64);
        let whole = data.len() & !3;
        if whole > 0 {
            self.queue.write_buffer(&buf, 0, &data[..whole]);
        }
        if whole < data.len() {
            let mut tail = data[whole..].to_vec();
            tail.resize(4, 0);
            self.queue.write_buffer(&buf, whole as u64, &tail);
        }
        buf
    }

    fn push(&self, c: Cmd) {
        let mut q = self.pending.lock().unwrap();
        q.push(c);
        if q.len() >= knobs().0 {
            let cmds = std::mem::take(&mut *q);
            drop(q);
            let _ = self.submit(cmds);
        }
    }

    fn run(&self, kernel: &'static str, p: &Params, src: Option<&wgpu::Buffer>, dst: &wgpu::Buffer, aux: Option<&wgpu::Buffer>,
           src8: Option<&wgpu::Buffer>, groups: (u32, u32)) {
        self.run_with(kernel, p, src, dst, aux, src8, groups, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with(&self, kernel: &'static str, p: &Params, src: Option<&wgpu::Buffer>, dst: &wgpu::Buffer, aux: Option<&wgpu::Buffer>,
                src8: Option<&wgpu::Buffer>, groups: (u32, u32), pipe: Option<wgpu::ComputePipeline>) {
        {
            let mut st = STATS.lock().unwrap();
            let e = st.get_or_insert_with(HashMap::new).entry(kernel).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += groups.0 as u64 * groups.1 as u64 * 256;
        }
        let hs = src.is_some_and(|b| self.is_half(b));
        let hd = self.is_half(dst);
        // a half picture is only ever made whole by a producer writing its own output (D / put mode 0): a kernel that
        // reads what is in dst, or adds into it, needs f32 there
        if hd && !(matches!(kernel, "project_mb" | "pm" | "from8" | "gdir" | "spread" | "shader")
                   || (matches!(kernel, "gauss" | "cubic" | "lanczos_v" | "project" | "mip") && p.u[13] == 0)) {
            panic!("kadr_core: {kernel} would write into a half-precision picture as if it were f32");
        }
        self.push(Cmd::Run { kernel, p: *p, src: src.cloned(), dst: dst.clone(), aux: aux.cloned(), src8: src8.cloned(), groups, hs, hd, pipe });
    }

    fn clear(&self, buf: &wgpu::Buffer) {
        self.push(Cmd::Clear(buf.clone()));
    }

    fn flush(&self) -> Option<wgpu::SubmissionIndex> {
        let cmds = std::mem::take(&mut *self.pending.lock().unwrap());
        if cmds.is_empty() {
            return None;
        }
        self.submit(cmds)
    }

    /// the recorded steps as one command buffer: all uniforms in one buffer, consecutive kernels in one compute pass
    fn submit(&self, cmds: Vec<Cmd>) -> Option<wgpu::SubmissionIndex> {
        if std::env::var_os("KADR_GPU_DRY").is_some() {
            let cmds: Vec<Cmd> = cmds.into_iter().filter(|c| matches!(c, Cmd::Copy(..))).collect();
            if cmds.is_empty() { return None; }
            let mut enc = self.device.create_command_encoder(&Default::default());
            for c in &cmds { if let Cmd::Copy(a, b, n) = c { enc.copy_buffer_to_buffer(a, 0, b, 0, *n); } }
            let i = self.queue.submit([enc.finish()]);
            self.wake();
            return Some(i);
        }
        let runs = cmds.iter().filter(|c| matches!(c, Cmd::Run { .. })).count();
        let mut ubytes = vec![0u8; (runs.max(1) as u64 * PSTRIDE) as usize];
        let mut k = 0usize;
        for c in &cmds {
            if let Cmd::Run { p, .. } = c {
                let at = k * PSTRIDE as usize;
                ubytes[at..at + std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
                k += 1;
            }
        }
        let ubuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &ubytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let d = &self.dummy;
        let ub = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &ubuf,
            offset: 0,
            size: std::num::NonZeroU64::new(std::mem::size_of::<Params>() as u64),
        });
        let mut groups_ = Vec::with_capacity(runs);
        for c in &cmds {
            if let Cmd::Run { src, dst, aux, src8, hs, hd, .. } = c {
                let (w, h16) = (&self.dummy_w, *hd);
                groups_.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ub.clone() },
                        wgpu::BindGroupEntry { binding: 1, resource: (if *hs { d } else { src.as_ref().unwrap_or(d) }).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: (if h16 { w } else { dst }).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: aux.as_ref().unwrap_or(d).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: src8.as_ref().unwrap_or(d).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: (if *hs { src.as_ref().unwrap_or(d) } else { d }).as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: (if h16 { dst } else { w }).as_entire_binding() },
                    ],
                }));
            }
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        if self.timing {
            return Some(self.timed(enc, &cmds, &groups_));
        }
        let mut k = 0usize;
        let mut i = 0usize;
        while i < cmds.len() {
            match &cmds[i] {
                Cmd::Run { .. } => {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    while i < cmds.len() {
                        let Cmd::Run { kernel, groups, hs, hd, pipe, .. } = &cmds[i] else { break };
                        match pipe {
                            Some(dp) => pass.set_pipeline(dp),
                            None => pass.set_pipeline(&self.pipe(kernel, *hs, *hd)),
                        }
                        pass.set_bind_group(0, &groups_[k], &[(k as u64 * PSTRIDE) as u32]);
                        pass.dispatch_workgroups(groups.0.max(1), groups.1.max(1), 1);
                        k += 1;
                        i += 1;
                    }
                }
                Cmd::Clear(b) => {
                    enc.clear_buffer(b, 0, None);
                    i += 1;
                }
                Cmd::Copy(a, b, n) => {
                    enc.copy_buffer_to_buffer(a, 0, b, 0, *n);
                    i += 1;
                }
            }
        }
        let i = self.queue.submit([enc.finish()]);
        self.wake();
        let cap = knobs().1;
        if cap > 0 {
            let old: Vec<wgpu::SubmissionIndex> = {
                let mut fl = self.inflight.lock().unwrap();
                fl.push_back(i.clone());
                let n = fl.len().saturating_sub(cap);
                fl.drain(..n).collect()
            };
            if let Some(last) = old.into_iter().last() {
                // waiting for the GPU lets go of Python: another render thread prepares its frame meanwhile. Every
                // submit runs inside a Python call holding the interpreter (never inside a detach), and no lock is
                // held here — a thread waiting for the interpreter while holding one would deadlock this one
                let py = unsafe { Python::assume_attached() };
                let dev = &self.device;
                py.detach(|| { let _ = dev.poll(wgpu::PollType::Wait { submission_index: Some(last), timeout: None }); });
            }
        }
        Some(i)
    }

    /// the steps one pass each between timestamps, waited for, their GPU time added to STATS per kernel
    fn timed(&self, mut enc: wgpu::CommandEncoder, cmds: &[Cmd], groups_: &[wgpu::BindGroup]) -> wgpu::SubmissionIndex {
        let runs = groups_.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: (2 * runs).max(2) });
        let mut k = 0u32;
        let mut names = Vec::new();
        for c in cmds {
            match c {
                Cmd::Run { kernel, groups, hs, hd, pipe, .. } => {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &qs, beginning_of_pass_write_index: Some(2 * k), end_of_pass_write_index: Some(2 * k + 1) }),
                    });
                    match pipe {
                        Some(dp) => pass.set_pipeline(dp),
                        None => pass.set_pipeline(&self.pipe(kernel, *hs, *hd)),
                    }
                    pass.set_bind_group(0, &groups_[k as usize], &[(k as u64 * PSTRIDE) as u32]);
                    pass.dispatch_workgroups(groups.0.max(1), groups.1.max(1), 1);
                    names.push(*kernel);
                    k += 1;
                }
                Cmd::Clear(b) => enc.clear_buffer(b, 0, None),
                Cmd::Copy(a, b, n) => enc.copy_buffer_to_buffer(a, 0, b, 0, *n),
            }
        }
        let bytes = (2 * runs.max(1)) as u64 * 8;
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let st = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        if runs > 0 {
            enc.resolve_query_set(&qs, 0..2 * runs, &rb, 0);
            enc.copy_buffer_to_buffer(&rb, 0, &st, 0, bytes);
        }
        let si = self.queue.submit([enc.finish()]);
        self.wake();
        if runs == 0 {
            return si;
        }
        let slice = st.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let v: Vec<u64> = slice.get_mapped_range().map(|m| bytemuck::cast_slice(&m).to_vec()).unwrap_or_default();
        let period = self.queue.get_timestamp_period() as f64;
        let mut stt = STATS.lock().unwrap();
        let map = stt.get_or_insert_with(HashMap::new);
        for (j, n) in names.iter().enumerate() {
            let dt = v.get(2 * j + 1).copied().unwrap_or(0).saturating_sub(v.get(2 * j).copied().unwrap_or(0));
            map.entry(n).or_insert((0, 0, 0)).2 += (dt as f64 * period) as u64;
        }
        si
    }

    fn grid(w: u32, h: u32) -> (u32, u32) {
        (w.div_ceil(16), h.div_ceil(16))
    }

    fn line(n: u32) -> (u32, u32) {
        let g = n.div_ceil(256);
        (g.min(65535), g.div_ceil(65535))
    }

    fn staging_buf(&self, bytes: u64) -> wgpu::Buffer {
        let mut st = self.staging.lock().unwrap();
        if let Some(k) = st.iter().position(|b| b.size() >= bytes) {
            return st.swap_remove(k);
        }
        drop(st);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// copy `bytes` of `buf` into a read-back buffer, send everything recorded and start mapping it — the GPU works
    /// on while the caller goes on; `finish` waits for that submission and takes the bytes
    fn start_read(&self, buf: &wgpu::Buffer, bytes: u64) -> (wgpu::Buffer, Option<wgpu::SubmissionIndex>) {
        let staging = self.staging_buf(bytes);
        self.push(Cmd::Copy(buf.clone(), staging.clone(), bytes));
        let si = self.flush();
        staging.slice(..bytes).map_async(wgpu::MapMode::Read, |_| {});
        (staging, si)
    }

    fn finish_read(&self, staging: wgpu::Buffer, si: Option<wgpu::SubmissionIndex>, bytes: u64) -> Vec<u8> {
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: si, timeout: None });
        let out = staging.slice(..bytes).get_mapped_range().map(|v| v.to_vec()).unwrap_or_default();
        staging.unmap();
        let mut st = self.staging.lock().unwrap();
        if st.len() < 4 {
            st.push(staging);
        }
        out
    }

    fn read(&self, buf: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let (st, si) = self.start_read(buf, bytes);
        self.finish_read(st, si, bytes)
    }
}

/// A picture on the GPU: w × h pixels of vec4<f32> (or packed RGBA8 for a source picture).
#[pyclass]
struct Img {
    g: &'static Gpu,            // the device (of the thread) that made it: its buffer goes back to that pool
    buf: wgpu::Buffer,
    half: bool,                 // 8 bytes a pixel (two packed f16 pairs) — a layer picture; canvases stay f32
    #[pyo3(get)]
    w: u32,
    #[pyo3(get)]
    h: u32,
    rgba8: bool,
    plane: bool,
    vpad: u32,                  // a source read as if padded by this many edge pixels on every side
    rgb24: bool,                // a source of packed RGB bytes (footage)
}

#[pymethods]
impl Img {
    /// back to numpy: (h, w, 4) float32 (a packed source: uint8)
    fn download<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = gpu()?;
        if self.rgba8 {                         // a packed source: its bytes (h, w, 4) uint8
            if self.rgb24 {
                return Err(PyRuntimeError::new_err("кадр видео RGB не скачивается"));
            }
            let n = self.w as u64 * self.h as u64 * 4;
            let mut bytes = g.read(&self.buf, n.next_multiple_of(4));
            bytes.truncate(n as usize);
            return Ok(PyArray1::from_vec(py, bytes).reshape([self.h as usize, self.w as usize, 4])?.into_any());
        }
        if self.half {
            let bytes = g.read(&self.buf, self.w as u64 * self.h as u64 * 8);
            let v: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|&x| f16_to_f32(x)).collect();
            return Ok(PyArray1::from_vec(py, v).reshape([self.h as usize, self.w as usize, 4])?.into_any());
        }
        let bytes = g.read(&self.buf, self.w as u64 * self.h as u64 * 16);
        let v: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        Ok(PyArray1::from_vec(py, v).reshape([self.h as usize, self.w as usize, 4])?.into_any())
    }
}

impl Drop for Img {
    fn drop(&mut self) {
        self.g.give_back(self.buf.clone());
    }
}

/// where a producer puts its result (see `put` in common.wgsl): into `to` at (ax, ay) — mode 1 added (motion blur),
/// mode 2 laid over with `opacity` and `clip`
fn put_into(p: &mut Params, to: &Img, ax: i32, ay: i32, mode: u32, opacity: f32, clip: Option<[f32; 4]>) {
    p.u[13] = mode; p.u[14] = to.w; p.u[15] = to.h; p.f[30] = ax as f32; p.f[31] = ay as f32;
    p.f[24] = opacity;
    if let Some(c) = clip {
        p.u[12] = 1;
        p.f[25..29].copy_from_slice(&c);
    }
}

/// a w × h layer picture in half precision (S / D in common.wgsl): half the memory traffic of every pass that reads it
fn img_h(g: &'static Gpu, w: u32, h: u32) -> Img {
    let buf = g.buffer(w as u64 * h as u64 * 8);
    g.half.lock().unwrap().insert(bkey(&buf));
    Img { g, half: true, buf, w, h, rgba8: false, plane: false, vpad: 0, rgb24: false }
}

fn bkey(b: &wgpu::Buffer) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    b.hash(&mut h);
    h.finish()
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let v = if e == 0 { m as f32 * (1.0 / 16777216.0) } else if e == 31 { if m == 0 { f32::INFINITY } else { f32::NAN } }
            else { f32::from_bits(((e + 112) << 23) | (m << 13)) };
    if s == 1 { -v } else { v }
}

fn img(g: &'static Gpu, w: u32, h: u32) -> Img {
    Img { g, half: false, buf: g.buffer(w as u64 * h as u64 * 16), w, h, rgba8: false, plane: false, vpad: 0, rgb24: false }
}

/// a flat shape with holes cut into triangles (ear clipping): `points` — x, y, x, y, …; `holes` — the number of the
/// point each hole begins with → three point numbers a triangle (kadr/extrude.py: the faces of 3D text)
#[pyfunction]
fn triangulate(points: Vec<f64>, holes: Vec<usize>) -> PyResult<Vec<u32>> {
    earcutr::earcut(&points, &holes, 2)
        .map(|v| v.into_iter().map(|i| i as u32).collect())
        .map_err(|e| PyRuntimeError::new_err(format!("triangulate: {e:?}")))
}

#[pyfunction]
fn gpu_info() -> PyResult<String> {
    Ok(gpu()?.name.clone())
}

/// a new picture filled with one value (rgb, a)
#[pyfunction]
fn new_img(w: u32, h: u32, value: [f32; 4]) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, w, h);
    let n = w * h;
    if value == [0.0; 4] {
        g.clear(&out.buf);
    } else {
        let mut p = Params::new();
        p.u[0] = n;
        p.f[..4].copy_from_slice(&value);
        g.run("fill", &p, None, &out.buf, None, None, Gpu::line(n));
    }
    Ok(out)
}

/// (h, w, 4) float32 → GPU; `half`: kept in f16 (a mip pyramid — read many times a pixel, half the traffic)
#[pyfunction]
#[pyo3(signature = (a, half=false))]
fn upload(a: PyReadonlyArray3<f32>, half: bool) -> PyResult<Img> {
    let g = gpu()?;
    let s = a.shape();
    if s[2] != 4 {
        return Err(PyRuntimeError::new_err("нужен массив (h, w, 4)"));
    }
    let data = a.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    if half {
        let h16: Vec<u16> = data.iter().map(|&x| f32_to_f16(x.clamp(-60000.0, 60000.0))).collect();
        let buf = g.upload(bytemuck::cast_slice(&h16));
        g.half.lock().unwrap().insert(bkey(&buf));
        return Ok(Img { g, half: true, buf, w: s[1] as u32, h: s[0] as u32, rgba8: false, plane: false, vpad: 0, rgb24: false });
    }
    Ok(Img { g, half: false, buf: g.upload(bytemuck::cast_slice(data)), w: s[1] as u32, h: s[0] as u32, rgba8: false, plane: false, vpad: 0, rgb24: false })
}

/// f32 → f16 bits, rounded to nearest even (as the GPU's pack2x16float), the value already within f16's range
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {                                 // subnormal or zero in f16
        if e < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let mut r = m >> shift;
        if rem > halfway || (rem == halfway && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let rem = m & 0x1fff;
    let mut r = ((e as u32) << 10) | (m >> 13);
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;                                 // a carry into the exponent is still the right number
    }
    sign | r.min(0x7bff) as u16
}

/// a source picture (h, w, 4) uint8 premultiplied (PIL «RGBa») → GPU, kept as packed RGBA8
#[pyfunction]
fn upload_rgba8(a: PyReadonlyArray3<u8>) -> PyResult<Img> {
    let g = gpu()?;
    let s = a.shape();
    let data = a.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    Ok(Img { g, half: false, buf: g.upload(data), w: s[1] as u32, h: s[0] as u32, rgba8: true, plane: false, vpad: 0, rgb24: false })
}

/// draw `src` (a layer's picture) onto the canvas `dst` at (x0, y0): opacity, clip rectangle, matte, blend mode, or a
/// single colour with src's alpha (a shadow)
#[pyfunction]
#[pyo3(signature = (dst, src, x0, y0, opacity, clip=None, matte=None, mx=0, my=0, mode=0, color=None, premul=0.0))]
#[allow(clippy::too_many_arguments)]
fn composite(dst: &Img, src: &Img, x0: i32, y0: i32, opacity: f32, clip: Option<[f32; 4]>, matte: Option<PyRef<Img>>,
             mx: i32, my: i32, mode: u32, color: Option<[f32; 3]>, premul: f32) -> PyResult<()> {
    let g = gpu()?;
    let mut p = Params::new();
    p.u[0] = dst.w; p.u[1] = dst.h; p.u[2] = src.w; p.u[3] = src.h; p.u[4] = mode;
    p.f[0] = x0 as f32; p.f[1] = y0 as f32; p.f[2] = opacity; p.f[12] = premul;
    if let Some(c) = clip {
        p.u[5] = 1;
        p.f[3..7].copy_from_slice(&c);
    }
    if let Some(c) = color {
        p.u[7] = 1;
        p.f[7..10].copy_from_slice(&c);
    }
    if let Some(m) = &matte {
        p.u[6] = 1; p.u[8] = m.w; p.u[9] = m.h; p.u[10] = if m.plane { 1 } else { 4 };
        p.f[10] = mx as f32; p.f[11] = my as f32;
    }
    let m32 = matte.as_ref().and_then(|m| as_f32(g, m));   // the matte is read as raw floats
    g.run("composite", &p, Some(&src.buf), &dst.buf, m32.as_ref().map(|m| &m.buf).or(matte.as_ref().map(|m| &m.buf)), None, Gpu::grid(src.w, src.h));
    Ok(())
}

fn gauss_weights(sigma: f32, d: f32) -> (Vec<f32>, i32) {
    let r = (3.0 * sigma + d.abs()).ceil() as i32 + 1;
    let s = sigma.max(1e-3) as f64;
    let mut w: Vec<f64> = (-r..=r).map(|k| (-0.5 * ((k as f64 - d as f64) / s).powi(2)).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.iter_mut().for_each(|x| *x /= sum);
    (w.into_iter().map(|x| x as f32).collect(), r)
}

/// one axis of the Gaussian into a new w × h picture; `inp`: the source and where it sits in that picture (a padding
/// that is never made — zeros around it)
fn gauss_pass(g: &'static Gpu, src: &Img, sigma: f32, d: f32, axis: u32, inp: (u32, u32, u32, u32), put: Option<PutTo>, half: bool) -> Option<Img> {
    let (w, h, px, py) = inp;
    let (wt, r) = gauss_weights(sigma, d);
    let wb = g.upload(bytemuck::cast_slice(&wt));
    let mut p = put.map(|(q, _, _)| *q).unwrap_or_else(Params::new);
    p.u[0] = w; p.u[1] = h; p.u[2] = axis; p.u[3] = r as u32;
    p.u[4] = src.w; p.u[5] = src.h; p.u[6] = px; p.u[7] = py;
    if let Some((_, to, cut)) = put {
        g.run("gauss", &p, Some(&src.buf), &to.buf, Some(&wb), cut, Gpu::grid(w, h));
        return None;
    }
    let out = if half { img_h(g, w, h) } else { img(g, w, h) };   // half: read only by the next pass
    g.run("gauss", &p, Some(&src.buf), &out.buf, Some(&wb), None, Gpu::grid(w, h));
    Some(out)
}

/// where the last pass of a blur puts its result: params already set up by put_into, the canvas, and the picture a
/// shadow is cut by (bound as src8)
type PutTo<'a> = (&'a Params, &'a Img, Option<&'a wgpu::Buffer>);

/// a Gaussian of σ centred at the fractional offset (dx, dy) over all four channels, zero outside; sx/sy: σ per axis
/// (a directional blur when they differ)
#[pyfunction]
#[pyo3(signature = (src, sx, sy, dx=0.0, dy=0.0))]
fn gauss(src: &Img, sx: f32, sy: f32, dx: f32, dy: f32) -> PyResult<Img> {
    let g = gpu()?;
    Ok(blur(g, src, sx, sy, dx, dy, (src.w, src.h, 0, 0), None).expect("a picture"))
}

/// the blur behind `gauss` and `shadow`: `inp` — the output size and where the source sits in it; `put` — the last
/// pass lays the result straight onto a canvas (put() in common.wgsl) instead of making a picture
#[allow(clippy::too_many_arguments)]
fn blur(g: &'static Gpu, src: &Img, sx: f32, sy: f32, dx: f32, dy: f32, inp: (u32, u32, u32, u32), put: Option<PutTo>) -> Option<Img> {
    let (w, h, _, _) = inp;
    if sx == sy && sx > 6.0 {
        return gauss_wide(g, src, sx, dx, dy, inp, put);
    }
    let a = gauss_pass(g, src, sx, dx, 0, inp, None, true).expect("a picture");
    gauss_pass(g, &a, sy, dy, 1, (w, h, 0, 0), put, false)
}

/// a wide Gaussian the way render.py gauss_shift does it: the picture q× smaller (box mean), blurred there by the
/// σ that is left after the box, and brought back up with Pillow's bicubic — nothing fine survives a wide blur anyway
fn gauss_wide(g: &'static Gpu, src: &Img, sigma: f32, dx: f32, dy: f32, inp: (u32, u32, u32, u32), put: Option<PutTo>) -> Option<Img> {
    let (w, h, px, py) = inp;
    let q = ((sigma / 3.0).floor() as u32).min(8).max(1);
    let (sw, sh) = (w.div_ceil(q), h.div_ceil(q));
    let small = img(g, sw, sh);
    let mut p = Params::new();
    p.u[0] = w; p.u[1] = h; p.u[2] = sw; p.u[3] = sh; p.u[4] = q;
    p.u[5] = src.w; p.u[6] = src.h; p.u[7] = px; p.u[8] = py;
    g.run("boxdown", &p, Some(&src.buf), &small.buf, None, None, Gpu::grid(sw, sh));
    let qf = q as f32;
    let s2 = ((sigma * sigma - (qf * qf - 1.0) / 12.0).max(0.25)).sqrt() / qf;
    let a = gauss_pass(g, &small, s2, dx / qf, 0, (sw, sh, 0, 0), None, true).expect("a picture");
    let b = gauss_pass(g, &a, s2, dy / qf, 1, (sw, sh, 0, 0), None, true).expect("a picture");
    let mid = img_h(g, w, sh);
    let mut p = Params::new();
    p.u[0] = sw; p.u[1] = sh; p.u[2] = w; p.u[3] = sh; p.u[4] = 0; p.f[0] = 1.0 / qf;
    g.run("cubic", &p, Some(&b.buf), &mid.buf, None, None, Gpu::grid(w, sh));
    let mut p = put.map(|(q, _, _)| *q).unwrap_or_else(Params::new);
    p.u[0] = w; p.u[1] = sh; p.u[2] = w; p.u[3] = h; p.u[4] = 1; p.f[0] = 1.0 / qf;
    if let Some((_, to, cut)) = put {
        g.run("cubic", &p, Some(&mid.buf), &to.buf, None, cut, Gpu::grid(w, h));
        return None;
    }
    let out = img(g, w, h);
    g.run("cubic", &p, Some(&mid.buf), &out.buf, None, None, Gpu::grid(w, h));
    Some(out)
}

/// a layer's drop shadow straight onto the canvas — render.py's padded → gauss → cut_under → composite in two or
/// three passes: the layer picture `im` (rgb, a) blurred by σ at (dx, dy) in a picture R px larger on every side
/// (never made), in `color`, kept outside the layer itself if `outside`, laid at (x0, y0) of the canvas with
/// `opacity` and `clip`
#[pyfunction]
#[pyo3(signature = (canvas, im, sigma, dx, dy, r, x0, y0, opacity, clip, color, outside, spread=0.0, cut=None, cx=0, cy=0))]
#[allow(clippy::too_many_arguments)]
fn shadow(canvas: &Img, im: &Img, sigma: f32, dx: f32, dy: f32, r: u32, x0: i32, y0: i32, opacity: f32, clip: Option<[f32; 4]>,
          color: [f32; 3], outside: bool, spread: f32, cut: Option<PyRef<Img>>, cx: i32, cy: i32) -> PyResult<()> {
    let g = gpu()?;
    // CSS spread: the alpha grown (shrunk) first, gp px wider on every side — the shadow moves out by gp, the cut not
    let grown = (spread.abs() >= 1e-6).then(|| spread_img(g, im, spread));
    let gp = if grown.is_some() { spread.abs().ceil() as u32 + 1 } else { 0 };
    let base = grown.as_ref().unwrap_or(im);
    let (w, h) = (base.w + 2 * r, base.h + 2 * r);
    let (x0, y0) = (x0 - gp as i32, y0 - gp as i32);
    let mut p = Params::new();
    put_into(&mut p, canvas, x0, y0, 3, opacity, clip);
    p.f[16..19].copy_from_slice(&color);
    // the cut: the layer itself (`cut`, else `im`) at (cx, cy) from im's corner — a 3D layer's shadow is a moved copy
    let cut_im: &Img = cut.as_deref().unwrap_or(im);
    if outside {
        p.u[9] = 1; p.u[10] = cut_im.w; p.u[8] = cut_im.h;
        p.f[19] = (r + gp) as f32 + cx as f32; p.f[20] = (r + gp) as f32 + cy as f32;
    }
    let im32 = if outside { as_f32(g, cut_im) } else { None };   // the cut reads the layer's alpha as raw floats
    blur(g, base, sigma, sigma, dx, dy, (w, h, r, r), Some((&p, canvas, outside.then_some(&im32.as_ref().unwrap_or(cut_im).buf))));
    Ok(())
}

/// lines of common.wgsl that come before a shader's own code (to give its author their own line numbers)
#[pyfunction]
fn shader_prefix_lines() -> usize {
    COMMON.lines().count() + 1
}

/// a shader from outside the crate (kadr/shader.py builds `code`: its prelude, the author's `effect`, `main`): one pass
/// into a new ow × oh picture. `src` — the layer's picture (S() in the kernel), `map` — a second picture as raw
/// floats in aux (a canvas: premultiplied 0..255, alpha 0..1); f (≤ 32) and u (≤ 16) fill the kernel's parameters as
/// they are. A mistake in the code raises with the compiler's message. `check`: compile only (kadr check)
#[pyfunction]
#[pyo3(signature = (key, code, ow, oh, f, u, src=None, map=None, half=true, check=false))]
#[allow(clippy::too_many_arguments)]
fn shader(key: &str, code: &str, ow: u32, oh: u32, f: Vec<f32>, u: Vec<u32>, src: Option<PyRef<Img>>, map: Option<PyRef<Img>>,
          half: bool, check: bool) -> PyResult<Option<Img>> {
    let g = gpu()?;
    if f.len() > 32 || u.len() > 16 {
        return Err(PyRuntimeError::new_err("shader: параметров больше, чем помещается (32 числа и 16 целых)"));
    }
    let hs = src.as_ref().is_some_and(|s| s.half);
    if check {
        for (a, b) in [(false, false), (true, true)] {
            g.dyn_pipe(key, code, a, b).map_err(PyRuntimeError::new_err)?;
        }
        return Ok(None);
    }
    let pipe = g.dyn_pipe(key, code, hs, half).map_err(PyRuntimeError::new_err)?;
    let m32 = map.as_ref().and_then(|m| as_f32(g, m));
    let out = if half { img_h(g, ow, oh) } else { img(g, ow, oh) };
    let mut p = Params::new();
    p.f[..f.len()].copy_from_slice(&f);
    p.u[..u.len()].copy_from_slice(&u);
    g.run_with("shader", &p, src.as_ref().map(|s| &s.buf), &out.buf, m32.as_ref().map(|m| &m.buf).or(map.as_ref().map(|m| &m.buf)),
               None, Gpu::grid(ow, oh), Some(pipe));
    Ok(Some(out))
}

/// one Gaussian pass of `sigma` px along the unit vector (ux, uy), the result moved by (sx, sy) — render.blur_dir;
/// the picture keeps its size (pad it first). `half`: the result in f16 (an inner pass)
#[pyfunction]
#[pyo3(signature = (src, sigma, ux, uy, sx=0.0, sy=0.0, half=false))]
fn gauss_dir(src: &Img, sigma: f32, ux: f32, uy: f32, sx: f32, sy: f32, half: bool) -> PyResult<Img> {
    let g = gpu()?;
    let (r, wt): (i32, Vec<f32>) = if sigma > 0.25 {
        let r = (3.0 * sigma as f64).ceil() as i32 + 1;
        let w: Vec<f64> = (-r..=r).map(|k| (-0.5 * (k as f64 / sigma as f64).powi(2)).exp()).collect();
        let s: f64 = w.iter().sum();
        (r, w.iter().map(|v| (v / s) as f32).collect())
    } else {
        (0, vec![1.0])
    };
    let wb = g.upload(bytemuck::cast_slice(&wt));
    let out = if half { img_h(g, src.w, src.h) } else { img(g, src.w, src.h) };
    let mut p = Params::new();
    p.u[0] = src.w; p.u[1] = src.h; p.u[2] = r as u32;
    p.f[0] = ux; p.f[1] = uy; p.f[2] = sx; p.f[3] = sy;
    g.run("gdir", &p, Some(&src.buf), &out.buf, Some(&wb), None, Gpu::grid(src.w, src.h));
    g.give_back(wb);
    Ok(out)
}

/// CSS box-shadow spread on a picture's alpha (render.spread_alpha): grown by r px through a disc with a soft rim,
/// r < 0 shrinks it; the result is ceil(|r|) + 1 px larger on every side, its colour unused (a shadow takes its own)
#[pyfunction]
fn spread(im: &Img, r: f32) -> PyResult<Img> {
    Ok(spread_img(gpu()?, im, r))
}

fn spread_img(g: &'static Gpu, im: &Img, r: f32) -> Img {
    let gp = r.abs().ceil() as u32 + 1;
    let out = img(g, im.w + 2 * gp, im.h + 2 * gp);
    let mut p = Params::new();
    p.u[0] = out.w; p.u[1] = out.h; p.u[2] = im.w; p.u[3] = im.h; p.u[4] = gp; p.u[5] = (r < 0.0) as u32;
    p.f[0] = r.abs();
    g.run("spread", &p, Some(&im.buf), &out.buf, None, None, Gpu::grid(out.w, out.h));
    out
}

/// (rgb, a) ↔ (rgb·a, a): dir 0 premultiplies, 1 divides back, 2 scales by `k` then divides (an average of samples)
#[pyfunction]
#[pyo3(signature = (src, dir, k=1.0))]
fn pm(src: &Img, dir: u32, k: f32) -> PyResult<Img> {
    let g = gpu()?;
    let out = img_h(g, src.w, src.h);
    let mut p = Params::new();
    let n = src.w * src.h;
    p.u[0] = n; p.u[1] = dir; p.f[0] = k;
    g.run("pm", &p, Some(&src.buf), &out.buf, None, None, Gpu::line(n));
    Ok(out)
}

/// a w × h picture of zeros with `src` placed at (x0, y0) (padding before a blur)
#[pyfunction]
fn padded(src: &Img, w: u32, h: u32, x0: i32, y0: i32) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, w, h);
    g.clear(&out.buf);
    blit_into(g, &out, src, x0, y0, 0);
    Ok(out)
}

/// the picture in f32 when it is half: for the bindings that read a picture's raw floats (aux, src8)
fn as_f32(g: &'static Gpu, im: &Img) -> Option<Img> {
    if !im.half {
        return None;
    }
    let out = img(g, im.w, im.h);
    blit_into(g, &out, im, 0, 0, 0);
    Some(out)
}

fn blit_into(g: &'static Gpu, dst: &Img, src: &Img, x0: i32, y0: i32, mode: u32) {
    let mut p = Params::new();
    p.u[0] = dst.w; p.u[1] = dst.h; p.u[2] = src.w; p.u[3] = src.h; p.u[4] = mode;
    p.f[0] = x0 as f32; p.f[1] = y0 as f32;
    g.run("blit", &p, Some(&src.buf), &dst.buf, None, None, Gpu::grid(src.w, src.h));
}

/// `src` laid into `dst` at (x0, y0) as it is, replacing what was there (a canvas rebuilt elsewhere and put back)
#[pyfunction]
#[pyo3(signature = (dst, src, x0=0, y0=0))]
fn put(dst: &Img, src: &Img, x0: i32, y0: i32) -> PyResult<()> {
    blit_into(gpu()?, dst, src, x0, y0, 0);
    Ok(())
}

/// add a sample's (rgb, a) premultiplied into the accumulator at (x0, y0) — motion blur
#[pyfunction]
#[pyo3(signature = (acc, src, x0, y0, premul=false))]
fn accumulate(acc: &Img, src: &Img, x0: i32, y0: i32, premul: bool) -> PyResult<()> {
    blit_into(gpu()?, acc, src, x0, y0, if premul { 4 } else { 1 });
    Ok(())
}

/// Pillow's `src.resize((ow, oh), LANCZOS, box)` of a padded RGBA8 picture, returned as a layer picture whose alpha
/// also carries the coverage of the layer's box `cover` (x0, y0, x1, y1 in output px) — render.py `_raster`. With `acc`:
/// added into that motion-blur accumulator at (ax, ay) instead, as `project` does
#[pyfunction]
#[pyo3(signature = (src, ow, oh, bx, cover, grade=None, acc=None, ax=0, ay=0, mode=1, opacity=1.0, clip=None))]
#[allow(clippy::too_many_arguments)]
fn lanczos(src: &Img, ow: u32, oh: u32, bx: [f32; 4], cover: [f32; 4], grade: Option<Vec<(u32, f32)>>,
           acc: Option<PyRef<Img>>, ax: i32, ay: i32, mode: u32, opacity: f32, clip: Option<[f32; 4]>) -> PyResult<Option<Img>> {
    let g = gpu()?;
    if !src.rgba8 {
        return Err(PyRuntimeError::new_err("lanczos берёт исходную картинку RGBA8"));
    }
    let (fx, fy) = ((bx[2] - bx[0]) / ow as f32, (bx[3] - bx[1]) / oh as f32);
    let (fsx, fsy) = (fx.max(1.0), fy.max(1.0));
    let (supx, supy) = (3.0 * fsx, 3.0 * fsy);
    // the source rows the vertical pass reads
    let y_lo = ((bx[1] + 0.5 * fy - supy + 0.5).floor() as i32).max(0);
    let y_hi = ((bx[1] + (oh as f32 - 0.5) * fy + supy + 0.5).floor() as i32).min((src.h + 2 * src.vpad) as i32);
    let nrows = (y_hi - y_lo).max(1) as u32;
    let tmp = img(g, ow, nrows);
    let mut p = Params::new();
    p.u[0] = src.w + 2 * src.vpad; p.u[1] = src.h + 2 * src.vpad; p.u[2] = ow; p.u[3] = y_lo as u32; p.u[4] = nrows;
    p.u[5] = src.vpad; p.u[6] = src.rgb24 as u32;
    if let Some(gr) = &grade {
        p.u[7] = gr.len().min(6) as u32;
        for (k, (c, v)) in gr.iter().take(6).enumerate() {
            p.u[8 + k] = *c;
            p.f[16 + k] = *v;
        }
    }
    p.f[0] = bx[0]; p.f[1] = fx; p.f[2] = fsx; p.f[3] = supx;
    let (wh, sh) = lanczos_weights(g, ow, p.u[0], bx[0], fx, fsx, supx);
    p.u[14] = sh;
    g.run("lanczos_h", &p, None, &tmp.buf, Some(&wh), Some(&src.buf), Gpu::grid(ow, nrows));
    g.give_back(wh);
    let mut q = Params::new();
    q.u[0] = ow; q.u[1] = oh; q.u[2] = nrows;
    q.f[0] = bx[1] - y_lo as f32; q.f[1] = fy; q.f[2] = fsy; q.f[3] = supy;
    q.f[4..8].copy_from_slice(&cover);
    let (wv, sv) = lanczos_weights(g, oh, nrows, bx[1] - y_lo as f32, fy, fsy, supy);
    q.u[3] = sv;
    let _keep = Defer(g, wv.clone());
    if let Some(a) = acc {                      // straight into an accumulator or onto a canvas
        put_into(&mut q, &a, ax, ay, mode, opacity, clip);
        g.run("lanczos_v", &q, Some(&tmp.buf), &a.buf, Some(&wv), None, Gpu::grid(ow, oh));
        return Ok(None);
    }
    let out = img_h(g, ow, oh);
    g.run("lanczos_v", &q, Some(&tmp.buf), &out.buf, Some(&wv), None, Gpu::grid(ow, oh));
    Ok(Some(out))
}

/// Lanczos weights for n outputs over n_in source samples (lzw.wgsl) — once per column / row, not per pixel;
/// returns the buffer and its stride in vec4
fn lanczos_weights(g: &'static Gpu, n: u32, n_in: u32, start: f32, scale: f32, fscale: f32, support: f32) -> (wgpu::Buffer, u32) {
    let k = (2.0 * support).ceil() as u32 + 2;
    let stride = 1 + k.div_ceil(4);
    let buf = g.buffer(n as u64 * stride as u64 * 16);
    let mut p = Params::new();
    p.u[0] = n; p.u[1] = n_in; p.u[2] = k; p.u[3] = stride;
    p.f[0] = start; p.f[1] = scale; p.f[2] = fscale; p.f[3] = support;
    g.run("lzw", &p, None, &buf, None, None, Gpu::grid(stride - 1, n));
    (buf, stride)
}

/// a buffer handed back to the pool when this goes out of scope
struct Defer<'a>(&'a Gpu, wgpu::Buffer);

impl Drop for Defer<'_> {
    fn drop(&mut self) {
        self.0.give_back(self.1.clone());
    }
}

/// the canvas → 8-bit RGBA (h, w, 4), alpha 255, with Kadr's fixed dither pattern `dither` (a plane)
#[pyfunction]
fn to_u8<'py>(py: Python<'py>, canvas: &Img, dither: &Img) -> PyResult<Bound<'py, PyAny>> {
    to_u8_start(canvas, dither)?.get(py)
}

/// a frame on its way back from the GPU (to_u8_start): `get()` waits for it — the CPU prepares the next frame meanwhile
#[pyclass]
struct Readback {
    staging: Option<wgpu::Buffer>,
    si: Option<wgpu::SubmissionIndex>,
    n: usize,
    w: usize,
    h: usize,
}

#[pymethods]
impl Readback {
    fn get<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = gpu()?;
        let st = self.staging.take().ok_or_else(|| PyRuntimeError::new_err("кадр уже забран"))?;
        let mut bytes = py.detach(|| g.finish_read(st, self.si.take(), (self.n.div_ceil(4) * 16) as u64));
        bytes.truncate(self.n * 4);
        Ok(PyArray1::from_vec(py, bytes).reshape([self.h, self.w, 4])?.into_any())
    }
}

/// to_u8 without waiting: the dither kernel and the copy back are sent now, the bytes are taken by `.get()`
#[pyfunction]
fn to_u8_start(canvas: &Img, dither: &Img) -> PyResult<Readback> {
    let g = gpu()?;
    let n = canvas.w * canvas.h;
    let quads = n.div_ceil(4);
    let out = g.buffer(quads as u64 * 16);
    let mut p = Params::new();
    p.u[0] = n;
    g.run("dither", &p, Some(&canvas.buf), &out, Some(&dither.buf), None, Gpu::line(quads));
    let (staging, si) = g.start_read(&out, quads as u64 * 16);
    g.give_back(out);
    Ok(Readback { staging: Some(staging), si, n: n as usize, w: canvas.w as usize, h: canvas.h as usize })
}

/// the canvas over nothing → 8-bit RGBA with straight colour and its alpha (export with transparency), not waited for:
/// `.get()` gives (h, w, 4). Was a float download and a numpy division on the CPU — ~2/3 of an alpha render's time
#[pyfunction]
fn to_rgba8_start(canvas: &Img) -> PyResult<Readback> {
    let g = gpu()?;
    let n = canvas.w * canvas.h;
    let quads = n.div_ceil(4);
    let out = g.buffer(quads as u64 * 16);
    let mut p = Params::new();
    p.u[0] = n;
    g.run("straight8", &p, Some(&canvas.buf), &out, None, None, Gpu::line(quads));
    let (staging, si) = g.start_read(&out, quads as u64 * 16);
    g.give_back(out);
    Ok(Readback { staging: Some(staging), si, n: n as usize, w: canvas.w as usize, h: canvas.h as usize })
}

/// a plain float array (h, w) — a matte or the dither pattern — on the GPU
#[pyfunction]
fn upload_plane(a: numpy::PyReadonlyArray2<f32>) -> PyResult<Img> {
    let g = gpu()?;
    let s = a.shape();
    let data = a.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    Ok(Img { g, half: false, buf: g.upload(bytemuck::cast_slice(data)), w: s[1] as u32, h: s[0] as u32, rgba8: false, plane: true, vpad: 0, rgb24: false })
}

/// the layer filter: ops [(code, value)] — 1 contrast 2 saturation 3 sepia 4 brightness 5 gamma — in order
#[pyfunction]
fn grade(src: &Img, ops: Vec<(u32, f32)>) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, src.w, src.h);
    let mut p = Params::new();
    let n = src.w * src.h;
    p.u[0] = n; p.u[1] = ops.len().min(14) as u32;
    for (k, (c, v)) in ops.iter().take(14).enumerate() {
        p.u[2 + k] = *c;
        p.f[k] = *v;
    }
    g.run("grade", &p, Some(&src.buf), &out.buf, None, None, Gpu::line(n));
    Ok(out)
}

/// a track matte from a canvas: alpha or luma, maybe inverted (into .x of a new image)
#[pyfunction]
fn matte(canvas: &Img, luma: bool, inverted: bool) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, canvas.w, canvas.h);
    let mut p = Params::new();
    let n = canvas.w * canvas.h;
    p.u[0] = n; p.u[1] = luma as u32; p.u[2] = inverted as u32;
    g.run("matte", &p, Some(&canvas.buf), &out.buf, None, None, Gpu::line(n));
    Ok(out)
}

/// a shadow kept only outside its layer: shadow.a ·= 1 − layer.a, the layer at (x0, y0) in the shadow's picture
#[pyfunction]
fn cut_under(shadow: &Img, layer: &Img, x0: i32, y0: i32) -> PyResult<()> {
    blit_into(gpu()?, shadow, layer, x0, y0, 2);
    Ok(())
}

/// a copy of a picture (a cached plate of still layers)
#[pyfunction]
fn copy(src: &Img) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, src.w, src.h);
    blit_into(g, &out, src, 0, 0, 0);
    Ok(out)
}

/// a Skia surface's premultiplied RGBA8 pixels (h, w, 4) uint8, ss× supersampled → a layer picture on the GPU
#[pyfunction]
fn from_skia(px: PyReadonlyArray3<u8>, ss: u32) -> PyResult<Img> {
    let g = gpu()?;
    let s = px.shape();
    let data = px.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    let src = g.upload(data);
    let (w, h) = (s[1] as u32 / ss, s[0] as u32 / ss);
    let out = img_h(g, w, h);
    let mut p = Params::new();
    p.u[0] = w; p.u[1] = h; p.u[2] = ss; p.u[3] = s[1] as u32;
    g.run("from8", &p, None, &out.buf, None, Some(&src), Gpu::grid(w, h));
    Ok(out)
}

/// a prepared picture (rgb, a) of content size cw × ch placed through the inverse matrix `inv` (raster px → content px,
/// row-major 3×3) into the output box x0, y0, x1, y1 (raster px) — bicubic Mitchell, anti-aliased box edges. With `acc`
/// the result is not kept: it is added into that motion-blur accumulator at (ax, ay), as `accumulate` would
#[pyfunction]
#[pyo3(signature = (pic, cw, ch, inv, bx, acc=None, ax=0, ay=0, mode=1, opacity=1.0, clip=None, crop=None))]
#[allow(clippy::too_many_arguments)]
fn project(pic: &Img, cw: f32, ch: f32, inv: [f32; 9], bx: [i32; 4], acc: Option<PyRef<Img>>, ax: i32, ay: i32, mode: u32,
           opacity: f32, clip: Option<[f32; 4]>, crop: Option<[f32; 4]>) -> PyResult<Option<Img>> {
    let g = gpu()?;
    let (ow, oh) = ((bx[2] - bx[0]).max(1) as u32, (bx[3] - bx[1]).max(1) as u32);
    let mut p = Params::new();
    p.u[0] = ow; p.u[1] = oh; p.u[2] = pic.w; p.u[3] = pic.h;
    p.f[..9].copy_from_slice(&inv);
    p.f[9] = cw; p.f[10] = ch; p.f[11] = bx[0] as f32; p.f[12] = bx[1] as f32;
    p.f[13..17].copy_from_slice(&crop.unwrap_or([0.0, 0.0, cw, ch]));
    if let Some(a) = acc {
        put_into(&mut p, &a, ax, ay, mode, opacity, clip);
        g.run("project", &p, Some(&pic.buf), &a.buf, None, None, Gpu::grid(ow, oh));
        return Ok(None);
    }
    let out = img_h(g, ow, oh);
    g.run("project", &p, Some(&pic.buf), &out.buf, None, None, Gpu::grid(ow, oh));
    Ok(Some(out))
}

/// a moving picture's motion blur in one pass (project_mb.wgsl): the same prepared picture (content cw × ch) through
/// every sample's inverse matrix, summed and × k (1 / samples drawn), into the output box — what project into an
/// accumulator sample by sample and then pm(acc, 2, k) made, without the accumulator
#[pyfunction]
#[pyo3(signature = (pic, cw, ch, invs, bx, k, solid=false, crop=None))]
#[allow(clippy::too_many_arguments)]
fn project_mb(pic: &Img, cw: f32, ch: f32, invs: Vec<[f32; 9]>, bx: [i32; 4], k: f32, solid: bool, crop: Option<[f32; 4]>) -> PyResult<Img> {
    let g = gpu()?;
    let (ow, oh) = ((bx[2] - bx[0]).max(1) as u32, (bx[3] - bx[1]).max(1) as u32);
    let mut flat = Vec::with_capacity(invs.len() * 12);
    for m in &invs {
        flat.extend_from_slice(m);
        flat.extend_from_slice(&[0.0; 3]);
    }
    let mb = g.upload(bytemuck::cast_slice(&flat));
    let out = img_h(g, ow, oh);
    let mut p = Params::new();
    p.u[0] = ow; p.u[1] = oh; p.u[2] = pic.w; p.u[3] = pic.h; p.u[4] = invs.len() as u32; p.u[5] = solid as u32;
    p.f[9] = cw; p.f[10] = ch; p.f[11] = bx[0] as f32; p.f[12] = bx[1] as f32; p.f[13] = k;
    p.f[14..18].copy_from_slice(&crop.unwrap_or([0.0, 0.0, cw, ch]));
    g.run("project_mb", &p, Some(&pic.buf), &out.buf, Some(&mb), None, Gpu::grid(ow, oh));
    g.give_back(mb);
    Ok(out)
}

/// a source picture of packed bytes: (h, w, 4) RGBA8 premultiplied or (h, w, 3) RGB (footage), read by `lanczos` as if
/// padded by `vpad` edge pixels — no copy, no repacking on the CPU
#[pyfunction]
#[pyo3(signature = (a, vpad=0))]
fn upload_source(a: PyReadonlyArray3<u8>, vpad: u32) -> PyResult<Img> {
    let g = gpu()?;
    let s = a.shape();
    let data = a.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    Ok(Img { g, half: false, buf: g.upload(data), w: s[1] as u32, h: s[0] as u32, rgba8: true, plane: false, vpad, rgb24: s[2] == 3 })
}

/// a canvas (a nested comp rendered on the GPU) → a packed RGBA8 premultiplied source picture, rounded as
/// render.py draw_comp rounds it — so it is placed like any picture
#[pyfunction]
fn canvas_to8(canvas: &Img) -> PyResult<Img> {
    let g = gpu()?;
    let n = canvas.w * canvas.h;
    let quads = n.div_ceil(4);
    let out = Img { g, half: false, buf: g.buffer(quads as u64 * 16), w: canvas.w, h: canvas.h, rgba8: true, plane: false, vpad: 0, rgb24: false };
    let mut p = Params::new();
    p.u[0] = n;
    g.run("to8", &p, Some(&canvas.buf), &out.buf, None, None, Gpu::line(quads));
    Ok(out)
}

/// a decoded picture file (h, w, 3|4) uint8, not premultiplied → a source picture (packed RGBA8 premultiplied), graded
/// [(code, value)] as render.apply_grade, mirrored, edge-padded by `pad` — load_image / load_plain on the GPU
#[pyfunction]
#[pyo3(signature = (a, grade=None, flip=false, pad=0))]
fn prep8(a: PyReadonlyArray3<u8>, grade: Option<Vec<(u32, f32)>>, flip: bool, pad: u32) -> PyResult<Img> {
    let g = gpu()?;
    let s = a.shape();
    if s[2] != 3 && s[2] != 4 {
        return Err(PyRuntimeError::new_err("нужен массив (h, w, 3) или (h, w, 4)"));
    }
    let data = a.as_slice().map_err(|_| PyRuntimeError::new_err("массив должен быть непрерывным"))?;
    let src = g.upload(data);
    let (w, h) = (s[1] as u32, s[0] as u32);
    let (ow, oh) = (w + 2 * pad, h + 2 * pad);
    let n = ow * oh;
    let quads = n.div_ceil(4);
    let out = Img { g, half: false, buf: g.buffer(quads as u64 * 16), w: ow, h: oh, rgba8: true, plane: false, vpad: 0, rgb24: false };
    let mut p = Params::new();
    p.u[0] = w; p.u[1] = h; p.u[2] = s[2] as u32; p.u[3] = flip as u32; p.u[4] = n; p.u[5] = pad;
    if let Some(gr) = &grade {
        p.u[7] = gr.len().min(6) as u32;
        for (k, (c, v)) in gr.iter().take(6).enumerate() {
            p.u[8 + k] = *c;
            p.f[16 + k] = *v;
        }
    }
    g.run("prep8", &p, None, &out.buf, None, Some(&src), Gpu::line(quads));
    Ok(out)
}

/// one colour (r, g, b) with a plane's values as alpha — a text from its coverage
#[pyfunction]
fn colorize(plane: &Img, color: [f32; 3]) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, plane.w, plane.h);
    let mut p = Params::new();
    let n = plane.w * plane.h;
    p.u[0] = n;
    p.f[..3].copy_from_slice(&color);
    g.run("colorize", &p, None, &out.buf, Some(&plane.buf), None, Gpu::line(n));
    Ok(out)
}

/// a gradient in its own box: out size, box (x0, y0, w, h relative to the output), kind 0 linear / 1 radial,
/// geometry [dx, dy, half, cx, cy, rx, ry], stops [(pos, r, g, b, a)]
#[pyfunction]
#[pyo3(signature = (ow, oh, bx, kind, geo, stops, premul=false))]
fn gradient(ow: u32, oh: u32, bx: [f32; 4], kind: u32, geo: [f32; 7], stops: Vec<[f32; 5]>, premul: bool) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, ow, oh);
    let flat: Vec<f32> = stops.iter().flatten().copied().collect();
    let sb = g.upload(bytemuck::cast_slice(&flat));
    let mut p = Params::new();
    p.u[0] = ow; p.u[1] = oh; p.u[2] = kind; p.u[3] = stops.len() as u32; p.u[4] = premul as u32;
    p.f[..4].copy_from_slice(&bx);
    p.f[4..11].copy_from_slice(&geo);
    g.run("gradient", &p, None, &out.buf, Some(&sb), None, Gpu::grid(ow, oh));
    Ok(out)
}

/// Motion Tile with mirror edges on a frame-sized picture: the tile x0, y0, x1, y1 (raster px) repeats around itself
#[pyfunction]
fn motion_tile(src: &Img, tile: [i32; 4], mirror: bool) -> PyResult<Img> {
    let g = gpu()?;
    let out = img(g, src.w, src.h);
    let mut p = Params::new();
    p.u[0] = src.w; p.u[1] = src.h; p.u[2] = mirror as u32;
    for k in 0..4 { p.f[k] = tile[k] as f32; }
    g.run("tile", &p, Some(&src.buf), &out.buf, None, None, Gpu::grid(src.w, src.h));
    Ok(out)
}

/// a picture's mip pyramid (`pyr`: the levels of kadr/mip.py `pyramid` one after another, level 0 w0 × h0, `levels`
/// of them) through one or more inverse matrices into the output box (mip.wgsl): anisotropic, trilinear, the content
/// box cw × ch by exact coverage, `tmap` = texel scale and origin, `wrap` 0 clamp / 1 repeat / 2 mirror. With `acc`
/// (one matrix only) the result is put there — mode 1 added, 2 laid over — instead of kept
#[pyfunction]
#[pyo3(signature = (pyr, w0, h0, levels, cw, ch, invs, bx, tmap, wrap, acc=None, ax=0, ay=0, mode=1, opacity=1.0, clip=None, k=None))]
#[allow(clippy::too_many_arguments)]
fn mip(pyr: &Img, w0: u32, h0: u32, levels: u32, cw: f32, ch: f32, invs: Vec<[f32; 9]>, bx: [i32; 4], tmap: [f32; 4], wrap: u32,
       acc: Option<PyRef<Img>>, ax: i32, ay: i32, mode: u32, opacity: f32, clip: Option<[f32; 4]>, k: Option<f32>) -> PyResult<Option<Img>> {
    let g = gpu()?;
    if invs.is_empty() {
        return Err(PyRuntimeError::new_err("mip: нет ни одной матрицы"));
    }
    let (ow, oh) = ((bx[2] - bx[0]).max(1) as u32, (bx[3] - bx[1]).max(1) as u32);
    let mut flat = Vec::with_capacity(invs.len() * 12);
    for m in &invs {
        flat.extend_from_slice(m);
        flat.extend_from_slice(&[0.0; 3]);
    }
    let mats = g.upload(bytemuck::cast_slice(&flat));
    let mut p = Params::new();
    p.u[0] = ow; p.u[1] = oh; p.u[2] = w0; p.u[3] = h0; p.u[4] = invs.len() as u32; p.u[5] = wrap; p.u[6] = levels;
    p.f[9] = cw; p.f[10] = ch; p.f[11] = bx[0] as f32; p.f[12] = bx[1] as f32; p.f[13] = k.unwrap_or(1.0 / invs.len() as f32);
    p.f[14..18].copy_from_slice(&tmap);
    if let Some(a) = acc {
        put_into(&mut p, &a, ax, ay, mode, opacity, clip);
        g.run("mip", &p, Some(&pyr.buf), &a.buf, Some(&mats), None, Gpu::grid(ow, oh));
        g.give_back(mats);
        return Ok(None);
    }
    let out = img_h(g, ow, oh);
    g.run("mip", &p, Some(&pyr.buf), &out.buf, Some(&mats), None, Gpu::grid(ow, oh));
    g.give_back(mats);
    Ok(Some(out))
}

/// send everything recorded so far to the GPU (with KADR_GPU_TIMING its times are then in `stats`)
#[pyfunction]
fn flush() -> PyResult<()> {
    gpu()?.flush();
    Ok(())
}

/// {kernel: (steps, threads, GPU nanoseconds — with KADR_GPU_TIMING only)} since the last call, then zeroed
#[pyfunction]
fn stats() -> HashMap<String, (u64, u64, u64)> {
    STATS.lock().unwrap().take().unwrap_or_default().into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

#[pymodule]
fn kadr_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Img>()?;
    m.add_class::<Readback>()?;
    m.add_class::<scene3d::Scene3d>()?;
    for f in [wrap_pyfunction!(gpu_info, m)?, wrap_pyfunction!(triangulate, m)?, wrap_pyfunction!(new_img, m)?, wrap_pyfunction!(upload, m)?,
              wrap_pyfunction!(upload_rgba8, m)?, wrap_pyfunction!(upload_plane, m)?, wrap_pyfunction!(composite, m)?,
              wrap_pyfunction!(gauss, m)?, wrap_pyfunction!(spread, m)?, wrap_pyfunction!(gauss_dir, m)?, wrap_pyfunction!(shader, m)?, wrap_pyfunction!(put, m)?, wrap_pyfunction!(shader_prefix_lines, m)?, wrap_pyfunction!(pm, m)?, wrap_pyfunction!(padded, m)?,
              wrap_pyfunction!(accumulate, m)?, wrap_pyfunction!(lanczos, m)?, wrap_pyfunction!(to_u8, m)?,
              wrap_pyfunction!(grade, m)?, wrap_pyfunction!(matte, m)?, wrap_pyfunction!(cut_under, m)?, wrap_pyfunction!(copy, m)?,
              wrap_pyfunction!(from_skia, m)?, wrap_pyfunction!(project, m)?,
              wrap_pyfunction!(upload_source, m)?, wrap_pyfunction!(colorize, m)?, wrap_pyfunction!(gradient, m)?,
              wrap_pyfunction!(motion_tile, m)?, wrap_pyfunction!(stats, m)?, wrap_pyfunction!(prep8, m)?, wrap_pyfunction!(to_u8_start, m)?, wrap_pyfunction!(to_rgba8_start, m)?,wrap_pyfunction!(shadow, m)?, wrap_pyfunction!(flush, m)?, wrap_pyfunction!(project_mb, m)?, wrap_pyfunction!(canvas_to8, m)?, wrap_pyfunction!(mip, m)?,
              wrap_pyfunction!(text::ft_setup, m)?, wrap_pyfunction!(text::glyph_cover, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
