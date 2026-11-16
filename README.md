# Kadr

A video engine for AI agents. The agent writes a **film document** (YAML): what is on screen, when, how it moves.
Kadr renders it on the GPU and answers in words — what is wrong with the film (`kadr check`) — and in pictures
(`kadr frame`, `kadr sheet`, `kadr scene`), so an agent that cannot watch a video can still see its work.

The aim: a person has the idea, the model does everything else. Kadr covers what is usually done in After Effects
and its plugins — keys and curves, parents, a 2.5D camera, motion blur, text on a path, shape morphing, masks and
mattes, time remapping, colour, glow, distortions, particles, a real 3D scene with lights and depth of field, block
characters with poses, tracking, stabilisation, cut-outs, sound-driven motion, charts. The list with a small film for
each technique is in [AE.md](AE.md) and `tests/ae/`.

> Status: young. Developed and tested on Apple-silicon macOS only. See «Platforms» below.
> The documents for the agent (`skill/SKILL.md`, `DESIGN.md`, `AE.md`) and Kadr's own messages are in Russian.

## Install

You need Python 3.10+ (3.13 is what it is tested on), a Rust toolchain (1.87+, for the render core) and `ffmpeg` /
`ffprobe` on the PATH.

```bash
git clone <this repository> kadr && cd kadr
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[serve]"                                 # the engine; [serve] adds the MCP server
pip install maturin && (cd core && maturin develop --release)     # the GPU core (kadr_core), a few minutes
kadr selftest                                             # the «no shimmer» guarantee on this machine
python tests/all.py                                       # everything, about three minutes
```

If your project lives in a cloud-synced folder, keep the virtual environment outside it.

## First film

```bash
kadr check  examples/hello/film.yaml                 # the report: what is wrong, in words
kadr frame  examples/hello/film.yaml --at 2.2 -o frame.png --gpu
kadr sheet  examples/hello/film.yaml --at 0.5 1.5 3 -o sheet.png --gpu
kadr render examples/hello/film.yaml --gpu -o hello.mp4
kadr scene  examples/figures/scene.yaml -o views.jpg       # a 3D scene from six sides, and what cannot be right in it
kadr catalog effects                                       # everything a document may say
```

- `examples/hello/` — the smallest film: layers, keys, a formula.
- `examples/figures/` — 3D: four block characters from a few words each, directed by poses.
- `tests/ae/*.yaml` — one small film per technique: the gallery and the test suite at once.

## For an agent

Give your agent [`skill/SKILL.md`](skill/SKILL.md): how to write a film, the work cycle (write → `check` → look at
frames → fix), every layer and effect. With Claude Code, copy the `skill` folder into your skills.

Kadr is also an MCP server — the renderer stays warm between calls, frames come back as pictures:

```json
{"mcpServers": {"kadr": {"command": "/path/to/.venv/bin/kadr", "args": ["serve"]}}}
```

Channel looks («styles») are plugins: Python files in `~/.kadr/plugins` that turn one line of a film («a title», «a
quote») into layers — see the head of `kadr/plugins.py` and `kadr/styles/base.py`. None ships with Kadr: a film is
written in layers until you make your own.

## Security

A film is a document, not a program: opening one from a stranger must not run code, read your files or reach the
network. Formulas are arithmetic only; a film reads its own folder; Python beside a film is not run — until you say
the folder is yours: `kadr trust <folder>`. Trust the folder where your own work lives, once. Details and what is
not protected: [SECURITY.md](SECURITY.md).

## Platforms

| | macOS (Apple silicon) | Windows, Linux |
|---|---|---|
| check, render, frame, sheet, scene, text, effects, 3D, sound | tested | **not tested yet.** The core is wgpu (Metal / D3D12 / Vulkan) and has no Apple-only code, the known breakages were fixed blind; expect rough edges (fonts by family name, reference frames) |
| `kadr matte`, `cutout`, `kadr reframe`, `kadr puppet --pose` | Apple Vision | not available: they use the Mac's own vision framework. Ready-made mattes play everywhere |
| `kadr render --hw` | hardware HEVC | falls back to x264 |
| `kadr depth` | yes | yes (PyTorch, in a Python of your choice: `KADR_ML_PYTHON`) |

## Contributing, licence

[CONTRIBUTING.md](CONTRIBUTING.md). MIT — [LICENSE](LICENSE). The bundled font Inter is under the SIL Open Font
License.
