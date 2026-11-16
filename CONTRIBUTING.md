# Contributing

Kadr is developed in the open, at the pace of its authors' own film work. Changes are welcome; nothing lands without
review.

- **Run the tests**: `python tests/all.py` (about three minutes on a laptop GPU). A change that alters pixels on
  purpose re-saves the frames it means to change (`python tests/ae/run.py save`) and says so.
- **One change, one pull request.** Say what it does and how you checked it.
- **A new technique comes with a small film** in `tests/ae/` that shows it, and a line in `AE.md`.
- **Security-sensitive places** — `kadr/safe.py`, the trust gate in `kadr/plugins.py` and `kadr/doc.py`
  (`may_read`, `resolve`), every call of ffmpeg — are changed only with a test in `tests/ae/test_safe.py`.
  A pull request must not add: `eval`/`exec` outside `kadr/safe.py`, `shell=True`, network access, new
  dependencies without a reason, files that are not text or small test media.
- **Dependencies and models** must allow commercial use (MIT, BSD, Apache-2.0, OFL for fonts).
- Code comments are in English; what Kadr says to its user (errors, reports) is in Russian — keep both.

The documents (`skill/SKILL.md`, `DESIGN.md`, `AE.md`) are in Russian: Kadr grew up there. Translations are welcome.
