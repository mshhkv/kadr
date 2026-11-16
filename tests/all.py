"""Every check of the engine in one go: the regression films (tests/reg) and each known-answer suite of tests/ae.
python tests/all.py [name …] — only the suites whose name has one of these words. One line a suite; exit 1 if any failed."""
import glob, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
want = [a for a in sys.argv[1:]]
suites = [('reg', [PY, os.path.join(HERE, 'reg', 'reg.py'), 'cmp'])] + [(os.path.basename(p)[:-3], [PY, p]) for p in [os.path.join(HERE, 'ae', 'run.py')] + sorted(glob.glob(os.path.join(HERE, 'ae', 'test_*.py')))]
bad, t_all = [], time.time()
for name, cmd in suites:
    if want and not any(w in name for w in want):
        continue
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    oks, fails = sum(ln.startswith('OK') for ln in lines), [ln for ln in lines if ln.startswith('FAIL')]
    last = lines[-1] if lines else ''
    good = r.returncode == 0 and not fails
    print(f'{"OK  " if good else "FAIL"} {name:18s} {time.time() - t0:6.1f} с  {last[:60]}' + (f'  ({oks} проверок)' if oks else ''))
    for ln in fails[:6]:
        print('       ' + ln[:200])
    if not good:
        bad.append(name)
        if not fails:
            print('       ' + '\n       '.join(lines[-4:])[:800])
print(f'{"ВСЁ СХОДИТСЯ" if not bad else "НЕ ПРОШЛИ: " + ", ".join(bad)} — {time.time() - t_all:.0f} с')
sys.exit(1 if bad else 0)
