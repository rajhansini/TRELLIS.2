"""judge_vqa.py -- Q1 (effect fidelity) as prompt-free VQA, per Sining's protocol.

HIS PROTOCOL, AND THE ONE DEVIATION
    His version: feed the text prompt to an LLM, get 5 yes/no questions, have a VLM
    answer them on each render, score = % yes.
    Ours: the questions are generated from the GT DRIVING VIDEO, not from the text
    prompt. Forced, because 22 of the 42 objects have no Kling prompt on disk
    (checked against KLING_PROMPTS_*.md: 4 exact matches, 16 mesh-token only,
    22 absent). It is also the stronger reference: the driving video is what we
    trained against and what the paper claims to match, whereas the text prompt is
    only what we asked Kling for, not what Kling produced.

TWO STAGES, DELIBERATELY SPLIT
    A. generate  -- one call per object against its GT video, writes questions.json.
       Frozen on disk and human-readable ON PURPOSE, so the questions can be audited
       and signed off BEFORE any scoring runs. Regenerating is opt-in (--regen).
    B. answer    -- one call per (object, method, view, repeat) returning all 5
       answers at once, so 42x6x4x3 costs ~3k calls rather than ~15k.

CALIBRATION, AND WHY IT IS NOT OPTIONAL
    The questions are written from the GT video, so GT answered against its own
    questions MUST score ~100%. If it does not, the questions are unanswerable or
    the VLM cannot see what was asked, and no other row means anything. Checked by
    build_vqa_table.py before it prints a single score.

QUESTION DESIGN
    A generator left to itself writes five variants of "is there lava?", which every
    method producing orange texture passes. The prompt therefore demands one question
    from each of five distinct axes (presence, appearance, spatial behaviour, temporal
    progression, restraint) and bans questions about the object's identity, the
    background and the camera, none of which any method gets wrong.

Usage
    python3 jobs/judge_vqa.py generate --objects spot_lava,chair_moss --out out/JUDGE/questions.json
    python3 jobs/judge_vqa.py answer --questions out/JUDGE/questions.json --all \
        --repeats 3 --out out/JUDGE/vqa.jsonl
"""
import argparse
import base64
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDS = Path('/net/projects/ranalab/rajhansini/baselines4d/_vids')
ENV = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh/.env.judge')
METHODS = ['frozen', 'meshnca', 'l4gm', 'sv4d2', 'dg4d', 'ours']
VIEWS = ['train', 'diagA', 'diagB', 'diagC']
MODEL = 'gemini-3.6-flash'

GEN_PROMPT = """This video is the ground truth for a surface-effect task: a fixed camera
films a single object whose SHAPE never changes while its SURFACE APPEARANCE evolves
over time.

Write exactly 5 yes/no questions that together verify whether another video reproduces
this same surface effect. The questions will be answered about other systems' attempts,
some of which reproduce the effect well and some badly, so they must be able to
distinguish between them.

Write exactly one question from each of these five axes:
  1. PRESENCE      -- is the defining material or pattern of the effect there at all?
  2. APPEARANCE    -- does it have the right colour, texture or finish? Name the specific
                      quality you observe, do not just repeat the material name.
  3. SPATIAL       -- is it in the right place on the object, or covering the right
                      amount of it?
  4. TEMPORAL      -- does it change over the clip the way it changes here? Name the
                      specific progression you observe.
  5. RESTRAINT     -- is the object free of a SURFACE failure this effect invites?
                      Pick something really visible in this clip staying absent.
                      This question is about the surface effect ONLY. It must NEVER
                      mention the object's shape, geometry, silhouette, proportions,
                      or the object melting, warping, deforming or dissolving. A
                      separate question elsewhere already scores shape, and asking
                      about it here would count the same failure twice. Ask instead
                      about how the effect itself misbehaves: covering too much,
                      washing out, saturating, bleeding past the surface, or losing
                      the contrast that makes it readable.

Rules, all of them binding:
  - Each question must be answerable YES or NO by looking at a video. Nothing subjective,
    nothing requiring measurement.
  - YES must always mean the effect was reproduced CORRECTLY. Never write a question
    where yes means failure.
  - Describe what you actually see in THIS video. Be concrete and specific.
  - Never ask about the object's identity, its shape, the background, the lighting, the
    camera, the resolution or the video quality. Only the surface effect.
  - Never mention a frame number, a timestamp or a duration.

Return JSON: {"effect": "<the effect in under 10 words>", "questions": [5 strings]}"""

ANSWER_PROMPT = """This video shows a 3D object whose surface appearance changes over time.

Answer each question below YES or NO based only on what you see in this video.
Answer about the SURFACE EFFECT only. Ignore the object's resolution, framing, playback
speed, and how long the clip is. If a question cannot be verified from this video,
answer NO.

{questions}

Return JSON: {{"answers": [<one true/false per question, in order>]}}"""

GEN_SCHEMA = {'type': 'object',
              'properties': {'effect': {'type': 'string'},
                             'questions': {'type': 'array', 'items': {'type': 'string'}}},
              'required': ['effect', 'questions']}
ANS_SCHEMA = {'type': 'object',
              'properties': {'answers': {'type': 'array', 'items': {'type': 'boolean'}}},
              'required': ['answers']}

_lock = threading.Lock()      # file + counters
_plock = threading.Lock()     # stdout ONLY. threading.Lock is not reentrant,
#                               and log() inside the _lock block self-deadlocked
#                               every worker at the first 50-record tick.


def log(*a):
    with _plock:
        print(*a, flush=True)


def load_key():
    for line in ENV.read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):
            return line.split('=', 1)[1].strip()
    sys.exit(f'no GEMINI_API_KEY in {ENV}')


def vpath(obj, method, view):
    return VIDS / (f'{obj}__{method}.mp4' if view == 'train'
                   else f'{obj}__{view}__{method}.mp4')


def part(path, fps=10):
    return {'inline_data': {'mime_type': 'video/mp4',
                            'data': base64.b64encode(path.read_bytes()).decode()},
            'video_metadata': {'fps': fps}}


def call(key, parts, prompt, schema, temp=1.0, timeout=300):
    body = {'contents': [{'parts': parts + [{'text': prompt}]}],
            'generationConfig': {'responseMimeType': 'application/json',
                                 'responseSchema': schema,
                                 'mediaResolution': 'MEDIA_RESOLUTION_LOW',
                                 'temperature': temp}}
    req = urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    out = json.loads(r['candidates'][0]['content']['parts'][0]['text'])
    return out, r.get('usageMetadata', {})


def retrying(fn, what, attempts=6):
    for i in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < attempts - 1:
                time.sleep(2 ** i + random.random() * 2)
                continue
            log(f'  FAIL {e.code} {what}: {e.read().decode()[:120]}')
            return None
        except Exception as e:                                    # noqa: BLE001
            if i < attempts - 1:
                time.sleep(2 ** i + random.random() * 2)
                continue
            log(f'  FAIL {what}: {type(e).__name__} {e}')
            return None


# ---------------------------------------------------------------- stage A
def generate(a, key):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(out.read_text()) if out.exists() else {}
    todo = [o for o in a.objects if a.regen or o not in store]
    log(f'generating questions for {len(todo)} objects ({len(store)} already frozen)')

    def one(obj):
        gt = vpath(obj, 'gt', 'train')
        if not gt.exists():
            log(f'  SKIP {obj}: no GT video'); return
        r = retrying(lambda: call(key, [part(gt)], GEN_PROMPT, GEN_SCHEMA, temp=0.4),
                     f'generate/{obj}')
        if not r:
            return
        d, _ = r
        qs = d.get('questions', [])
        if len(qs) != 5:
            log(f'  WARN {obj}: got {len(qs)} questions, not 5'); return
        with _lock:
            store[obj] = {'effect': d.get('effect', ''), 'questions': qs}
            out.write_text(json.dumps(store, indent=1, sort_keys=True))
        log(f'  {obj}  [{d.get("effect","")}]')

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    log(f'\nwrote {out}  ({len(store)} objects)')
    log('REVIEW THESE BEFORE SCORING. Nothing downstream is meaningful if a question '
        'is unanswerable or if YES does not mean success.')


# ---------------------------------------------------------------- stage B
def answer(a, key):
    qs = json.loads(Path(a.questions).read_text())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r['obj'], r['method'], r['view'], r['repeat']))
            except Exception:                                      # noqa: BLE001
                pass

    units = []
    for obj in a.objects:
        if obj not in qs:
            log(f'  SKIP {obj}: no frozen questions'); continue
        for rep in range(a.repeats):
            # gt is scored too, at its own view: it is the calibration row and must
            # come back at ~100%, since the questions were written from it.
            for m in METHODS + ['gt']:
                for v in (VIEWS if m != 'gt' else ['train']):
                    if vpath(obj, m, v).exists() and (obj, m, v, rep) not in done:
                        units.append((obj, m, v, rep))
    random.Random(0).shuffle(units)
    log(f'{len(done)} already done, {len(units)} to run, {a.workers} workers')

    state = {'n': 0, 'tok': 0}
    fh = out.open('a')

    def one(u):
        obj, m, v, rep = u
        block = '\n'.join(f'{i+1}. {q}' for i, q in enumerate(qs[obj]['questions']))
        r = retrying(lambda: call(key, [part(vpath(obj, m, v))],
                                  ANSWER_PROMPT.format(questions=block), ANS_SCHEMA),
                     f'{obj}/{m}/{v}#{rep}')
        if not r:
            return
        d, usage = r
        ans = d.get('answers', [])
        if len(ans) != 5:
            log(f'  WARN {obj}/{m}/{v}#{rep}: {len(ans)} answers'); return
        rec = dict(obj=obj, method=m, view=v, repeat=rep,
                   answers=[bool(x) for x in ans],
                   pct=100.0 * sum(bool(x) for x in ans) / 5,
                   tokens=usage.get('totalTokenCount'))
        with _lock:
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            state['n'] += 1
            state['tok'] += usage.get('totalTokenCount') or 0
            if state['n'] % 50 == 0:
                log(f"  {state['n']}/{len(units)} done, {state['tok']/1e6:.2f}M tokens")

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, units))
    fh.close()
    log(f'done {state["n"]}/{len(units)} in {(time.time()-t0)/60:.1f} min, '
        f'{state["tok"]/1e6:.2f}M tokens')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['generate', 'answer'])
    p.add_argument('--objects', default='')
    p.add_argument('--all', action='store_true')
    p.add_argument('--questions', default='out/JUDGE/questions.json')
    p.add_argument('--out', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--regen', action='store_true')
    a = p.parse_args()

    all_objs = sorted({f.name.split('__')[0] for f in VIDS.glob('*.mp4')})
    a.objects = all_objs if a.all else [o for o in a.objects.split(',') if o]
    bad = [o for o in a.objects if o not in all_objs]
    if bad:
        sys.exit(f'unknown objects: {bad}')

    key = load_key()
    (generate if a.stage == 'generate' else answer)(a, key)


if __name__ == '__main__':
    main()
