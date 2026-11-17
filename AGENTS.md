# For the agent working with Kadr

You are probably an AI agent: someone gave you Kadr to make a film with, or asked you to set it up. This page is for
you. How to write a film is in [`skill/SKILL.md`](skill/SKILL.md); this page is about the engine itself.

## The engine is yours to change

Kadr is built for agents and mostly by them. When it stands in your way — **fix it in the code**, do not work around
it in the film:

- **A bug** (a wrong frame, a crash, a check that lies, a message that sends you the wrong way) — find the cause and
  fix it, with a test that would have caught it.
- **A tool you lack** (a layer, an effect, a property, a check, a command, a thing you had to do by hand or blind) —
  add it to the engine, so that the next film gets it by one line of the document.
- **Something missing on this system** (see «Windows and Linux» below) — install or write it before the first film.

A workaround in a film is paid for again in every film after it, by you and by every other agent. A fix in the
engine is paid for once. You have the source, the tests and the eyes (`kadr frame`, `kadr sheet`): you do not need
anyone's permission to improve the copy you work in.

What goes where:

| What you need | Where it belongs |
|---|---|
| The look of one film, of one channel | the document itself, a template, a plugin in `~/.kadr/plugins` — `DESIGN.md`, «Расширение без правки движка». Not the engine |
| A bug; a general ability any film could use; a check that would have caught your mistake | the engine: `kadr/`, `core/` |
| A trap you fell into that the engine cannot catch | a line in `skill/SKILL.md`, «Грабли» |

## How to change it

1. **Reproduce it small.** A document of a few layers that shows the fault or needs the new thing. It becomes the test.
2. **Find the place.**

   | | |
   |---|---|
   | `kadr/doc.py` | the document: what a layer may say, defaults, times and words, every error and warning about the document |
   | `kadr/render.py` | the reference renderer (CPU), text layout, fonts |
   | `kadr/gpu.py`, `core/src/` | the GPU renderer and its Rust core (wgpu) |
   | `kadr/effects.py`, `kadr/effects_gpu.py`, `kadr/shaders/` | effects, on both renderers |
   | `kadr/check.py`, `kadr/scenecheck.py` | what `kadr check` and `kadr scene` find wrong |
   | `kadr/cli.py`, `kadr/serve.py` | the commands and the MCP server |
   | `kadr/plugins.py`, `kadr/styles/base.py` | plugins: styles, elements, layer types, effects |
   | `kadr/safe.py` | what a formula may do — the security gate |
   | `tests/ae/` | one small film and one suite per technique |

3. **Change it the way the engine is written:** comments in English and about *why*; what Kadr says to its user
   (errors, warnings, reports) in Russian, in plain words, naming what to do next. A mistake in a document is
   answered with a message, never with a traceback.
4. **Both renderers.** A thing that draws must draw the same on the CPU reference (`kadr/render.py`) and on the GPU;
   the suites compare them.
5. **Test.** `python tests/all.py` (about five minutes; `python tests/all.py name` runs the suites whose name has that
   word). A new ability comes with a small film in `tests/ae/` and a line in `AE.md`. If you changed pixels on
   purpose, re-save the answers you meant to change (`python tests/ae/run.py save`, `python tests/reg/reg.py save`)
   and say so.
6. **Look at the frames.** Tests pass on things that look wrong. `kadr sheet` the film you changed it for.
7. **Write it down:** `skill/SKILL.md` if a film is written differently now, `kadr catalog` picks up new parameters by
   itself.
8. **Tell your user** what you changed in the engine and why, apart from the film. Commit it apart from the film too.

Traps of the work itself:

- **`kadr serve` keeps the code it started with.** After you change the engine, the MCP tools of a running server
  still answer with the old code — use the command line (`kadr check`, `kadr frame` …) until the server is restarted.
- **After a change in `core/`** build it again: `cd core && maturin develop --release`.
- **One GPU, one thread.** A renderer is used from the thread that made it (`kadr/gpu.py` refuses otherwise; the
  server keeps one thread for it). Do not render on the GPU from threads of your own.
- **Do not change a default that finished films rely on** — add an option. A film rendered last month must render
  the same today.
- **Several agents in one copy:** look at `git status` first. Uncommitted changes that are not yours mean someone is
  working there now — leave them a note (what you found, a small document that shows it) instead of editing under
  their hands.

What stays closed:

- **The security gates** — `kadr/safe.py`, the trust check in `kadr/plugins.py` and `kadr/doc.py` (`may_read`,
  `resolve`), the way ffmpeg is called — are changed only with a test in `tests/ae/test_safe.py` and only with your
  user knowing. Never loosen them to make a film render. `kadr trust` is the user's decision, not yours.
- **No network, no `eval`/`exec` outside `kadr/safe.py`, no `shell=True`.**
- **Dependencies and models** must allow commercial use (MIT, BSD, Apache-2.0; OFL for fonts). Read the licence of a
  model's *weights*, not only of its code: many good ones are non-commercial.
- **Nothing leaves the machine without the user's yes**: no push, no pull request, no issue, no upload. If your fix
  would help others, say so and offer to send it upstream ([CONTRIBUTING.md](CONTRIBUTING.md)); the user decides.

## Windows and Linux: do this first

Kadr was made and tested on Apple-silicon macOS **only**. On Windows and Linux nobody has run it yet. If you are the
first agent on such a machine, setting the engine up is your first task — before any film. Do it once, well, and
every film after it is ordinary work.

What is absent on purpose, and why:

| | State outside macOS | Why | What to do |
|---|---|---|---|
| Cut-outs: `kadr matte`, `cutout: true` | **no program to make them** (ready `*.matte.mp4` / `*.matte.png` play everywhere) | on a Mac this is Apple Vision, built into the system: nothing to download. Other systems have nothing like it built in, and Kadr does not ship a model it has not tested | write `kadr/matte_tool.py` — below |
| `kadr reframe` (horizontal → vertical, following the person) | the same program | it asks the same program where the people and faces are | the same file |
| `kadr puppet --pose` (joints of a person in a photo) | the same program | the same | the same file |
| `kadr render --hw` | falls back to x264, says so | the hardware encoder used is the Mac's (`hevc_videotoolbox`) | optional: add this machine's encoder (`hevc_nvenc`, `hevc_qsv`, `hevc_amf`) beside `HW` in `kadr/cli.py`; check the colours against x264 before trusting it |
| Reference answers of the tests | some will differ | `tests/reg` keeps exact hashes of the CPU renderer's frames (another processor, another build of a library — another last bit); `tests/ae/ref` keeps GPU frames, compared with a tolerance | see step 3 |
| Fonts | bundled Inter always works; system fonts by file name | families are found by file names in the system's font folders (`FONT_DIRS`, `kadr/render.py`) | add this system's folders and names if a font is not found |
| Everything else (render, check, frame, sheet, scene, text, effects, 3D, sound, depth) | **written to work, never run** | the core is wgpu (D3D12 / Vulkan), no Apple-only code; the known breakages were fixed blind | steps 1–3 |

The order:

1. **Build.** Python 3.10+, a Rust toolchain (on Windows with the MSVC build tools), `ffmpeg` and `ffprobe` on the
   PATH. Then the README's «Install». Keep the virtual environment out of cloud-synced folders.
2. **`kadr selftest`**, then `kadr check` / `kadr frame` / `kadr render` on `examples/hello/film.yaml`. Look at the
   frame. This is the first render of Kadr on this system: expect a path, a font or an ffmpeg option to break. Fix
   each in the engine (paths through `os.path`, never by hand; no `/tmp`; no assumptions about `nice`, signals,
   file locks), not around it.
3. **`python tests/all.py`.** For every failing suite decide which it is:
   - *the engine is wrong here* (a crash, a missing file, a visibly wrong frame) — fix the engine;
   - *the frame is right and only differs from the Mac's* — look at it first (`python tests/ae/run.py sheets` writes
     every test film's frames into the folder `sheets`; `kadr frame` for the films of `tests/reg`), and only then
     re-save the answers for this machine (`python tests/ae/run.py save`, `python tests/reg/reg.py save`). `reg`
     compares exact hashes, so expect it to differ on a first run; `ae` allows small differences, so a failure
     there is more likely real. Do not re-save what you have not looked at: that turns a bug into the expected
     answer.
   The checks that need the Mac's vision print «пропущено» or «SKIP» and pass.
4. **The matte program** — the one large missing piece. Everything that cuts out or finds people goes through
   `kadr.matte.tool()`, which starts a program and talks to it in bytes: frames in, mattes (or numbers) out. On a Mac
   the program is `kadr/matte.swift`. Elsewhere it is **`kadr/matte_tool.py`, which you write** (or any program named
   in `$KADR_MATTE_TOOL`; a `.py` is run by the Python named in `$KADR_ML_PYTHON` or `~/.kadr/ml_python`, so the
   models can live in their own environment, as for `kadr depth`).
   - The whole language is in the head of [`tests/ae/matte_tool_example.py`](tests/ae/matte_tool_example.py) — a
     working program that calls «the thing» whatever is bright. Copy it to `kadr/matte_tool.py` and replace `find()`
     with models. Three modes are asked for: mattes (`person`, `subject`, `soft`), `boxes` (where the people and the
     largest face are, a line a frame), `pose` (joints and face points).
   - Choose models your user may use commercially, run them locally (ONNX Runtime and MediaPipe are the usual
     runtimes; a person segmenter, a salient-object segmenter, a body-pose and a face detector cover all three modes).
     **Check each licence yourself, weights included, and tell the user what you chose and from where** — downloading
     a model is their decision.
   - Hold it to what the Mac's does: `python tests/all.py matte reframe puppet` — the suites then run the real checks
     (a thing drawn by Kadr is cut out by its true outline, more than 93 % of the area; a person-free clip writes no
     file). Then `kadr matte … --sheet look.jpg` on real footage of your user's kind, and look: hair, a dark jacket
     on a dark wall, hands.
   - `subject` with a point must follow the *same* thing from frame to frame; `soft` and `person` must not flicker
     between frames. A segmenter run frame by frame flickers — smooth it or use one that knows the frames before.
5. **Write down what you found** for the next agent: fix the README's «Platforms» table to what is now true on this
   system, add the traps to `skill/SKILL.md`. Tell your user what works, what does not, and what you installed.

Until step 4 is done, films still render: only *making* cut-outs, reframing and pose-finding are missing, and Kadr
says so in words when asked for them.
