"""
render_arm.py — render ANY rung's checkpoint at a fixed camera.

WHY THIS EXISTS
  render_rung27_orbit.py does `import rung27_selfattn_lora as R` and builds its
  LoRARegistry from that module. Rungs 27/28/29/30/32/34 all save the same
  300-tensor, rung27-shaped state_dict, so they load fine. Rungs 31 and 33 do NOT:
  their dual-branch registry carries a SECOND set of bundles plus a per-block gate.
  Measured, not assumed:

      rung27/28/29/30/32/34   300 tensors  to_q to_kv to_out sa_qkv sa_out
      rung31/33               510 tensors  + gates

  Two separate things therefore have to change for a dual-branch arm, and MISSING
  EITHER ONE still ends in "Unexpected key(s) in state_dict".

TWO BUGS THIS FILE EXISTS TO AVOID, BOTH FOUND BY ONE CANARY JOB

  1. WRONG MODULE. Fixed by redirecting the import, NOT by pre-importing.
     The first version imported the trainer here, before handing off. That breaks:
     render_rung27_orbit.py builds a FAKE argv for the trainer (--mesh, --gt-dir,
     --out-dir <scratch>, --resolution ...) precisely because these trainers parse
     sys.argv at module scope. Pre-importing meant the trainer parsed the RENDERER's
     argv instead, where `--res 518` prefix-matched rung31's `--resolution`
     (choices 512/1024) and argparse killed the job before a single frame.
     So: install a meta_path finder that maps the NAME `rung27_selfattn_lora` to the
     dual trainer's FILE, and let the renderer import it, with its own fake argv, at
     the moment it chooses.

  2. WRONG REGISTRY SHAPE. The renderer calls
         R.LoRARegistry(..., targets=..., active=..., alpha=...)
     with no `dual` argument, and LoRARegistry defaults to dual=False. That builds
     the 300-tensor spatial-only registry, which cannot accept a 510-tensor
     checkpoint no matter which module it came from. So after the redirected module
     executes, its LoRARegistry is replaced by a subclass that supplies dual/branch/
     gate_init from the run's own config.json.

  Nothing in render_rung27_orbit.py is edited. The redirect and the subclass live
  entirely here.

WIDE CONTEXT IS ALREADY HANDLED, DOWNSTREAM
  Rungs 32 and 34 trained with --context-window 3 (1029 -> 3087 tokens).
  render_rung27_orbit.py already rebuilds that stack itself -- see its "rung32: WIDE
  CONTEXT WINDOW" block, which mirrors the trainer's _stack over offsets [-1,0,1] and
  carries a GATE-window asserting the centre slice is frame f byte-for-byte. This
  wrapper must not second-guess it; it reads context_window only to report it.
"""
import argparse
import importlib.abc
import importlib.util
import json
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='absolute path to the run directory')
ap.add_argument('--tag', required=True)
ap.add_argument('--yaw0', type=float, required=True)
ap.add_argument('--elev', type=float, required=True)
ap.add_argument('--turns', type=float, default=0.0,
                help='0 = camera FIXED, which is what the flicker metric needs: '
                     'every change between frames is then texture, not camera')
ap.add_argument('--n-frames', type=int, required=True)
ap.add_argument('--res', type=int, default=518)
ap.add_argument('--ckpt', default='lora_best.pt')
A, rest = ap.parse_known_args()

RUN = Path(A.run)
cfg_path = RUN / 'config.json'
if not cfg_path.exists():
    sys.exit(f'FAILED: {cfg_path} does not exist — the run never wrote a config, '
             f'so its rung and mcfm mode are unknown. Refusing to guess.')
CFG = json.loads(cfg_path.read_text())

rung = CFG.get('rung')
# The five wide-context runs written before the rung32 chain was fixed claim 27.
# Trust context_window over the stamped rung: it is what the model actually saw.
if CFG.get('context_window') and rung == 27:
    rung = 32

DUAL_RUNGS = {31, 33}                     # 510 tensors, carry `gates`
IS_DUAL = rung in DUAL_RUNGS
TRAINER = 'rung31_dual_attn_lora' if IS_DUAL else 'rung27_selfattn_lora'

print(f'[ARM] run={RUN.name}', flush=True)
print(f'[ARM] rung={rung}  mcfm={CFG.get("mcfm")}  '
      f'context_window={CFG.get("context_window")}  '
      f'temporal_window={CFG.get("temporal_window")}  dual={IS_DUAL}', flush=True)
print(f'[ARM] registry from {TRAINER}', flush=True)


class _RedirectLoader(importlib.abc.Loader):
    """Execute the dual trainer's source under the name the renderer imports.

    exec_module runs the file exactly as Python would -- so the trainer parses
    whatever argv the RENDERER has set at that instant, which is the fake argv it
    built for this purpose. Only after that do we swap LoRARegistry.
    """

    def __init__(self, real):
        self._real = real

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        if not IS_DUAL:
            return
        base = module.LoRARegistry
        branch = CFG.get('branch', 'both')
        gate_init = CFG.get('gate_init', 0.0)

        class _DualRegistry(base):
            # The renderer calls LoRARegistry(...) with no dual/branch/gate_init,
            # so without this it builds the 300-tensor spatial-only registry and
            # load_state_dict rejects the checkpoint's `gates` keys.
            def __init__(self, *a, **kw):
                kw.setdefault('dual', True)
                kw.setdefault('branch', branch)
                kw.setdefault('gate_init', gate_init)
                super().__init__(*a, **kw)

        module.LoRARegistry = _DualRegistry
        print(f'[ARM] LoRARegistry -> dual=True branch={branch} '
              f'gate_init={gate_init} (510-tensor registry)', flush=True)


class _RedirectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, name, path):
        self._name, self._path = name, path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._name:
            return None
        spec = importlib.util.spec_from_file_location(fullname, self._path)
        spec.loader = _RedirectLoader(spec.loader)
        return spec


if IS_DUAL:
    sys.meta_path.insert(0, _RedirectFinder(
        'rung27_selfattn_lora', str(_HERE / f'{TRAINER}.py')))
    print('[ARM] import redirect installed: rung27_selfattn_lora -> '
          f'{TRAINER}.py', flush=True)

# The renderer parses THIS argv at its own module scope, then builds its own fake
# argv for the trainer import. Do not import the trainer here.
sys.argv = [
    'render_rung27_orbit.py',
    '--run', str(RUN),
    '--ckpt', A.ckpt,
    '--sweep', 'both',
    '--n-frames', str(A.n_frames),
    '--n-angles', str(A.n_frames),
    '--turns', str(A.turns),
    '--yaw0', str(A.yaw0),
    '--elev', str(A.elev),
    '--res', str(A.res),
    '--tag', A.tag,
] + rest

runpy.run_path(str(_HERE / 'render_rung27_orbit.py'), run_name='__main__')
