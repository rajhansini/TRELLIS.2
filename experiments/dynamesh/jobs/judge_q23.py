"""judge_q23.py -- Q2 (temporal coherence) and Q3 (geometry correctness).

Q1 lives in judge_vqa.py: it is per-object yes/no VQA generated from the driving
video, per Sining's protocol. Q2 and Q3 are one fixed question each, identical for
all 42 objects, so they stay here.

WHAT CHANGED FROM THE FIRST ATTEMPT, AND WHY
    The first run asked "is the shape identical THROUGHOUT the video". That measures
    temporal stability of shape, which a method that hallucinates a wrong-but-steady
    geometry satisfies perfectly. L4GM's silhouette is ~29% larger than the true mesh
    and it scored 4.65 out of 5. The question could not see the claim.

    Q3 now supplies the TRUE SHAPE as a reference image -- the input mesh rendered
    untextured at the same camera as the video -- and asks whether the object in the
    video is that shape. jobs/make_shape_refs.py produces those references and
    calibrates its camera against the pipeline's own render rather than assuming a
    convention (solved at 0.9938 IoU; every rival convention scored below 0.70).

    Q2 is unchanged. It was the one reference-free question that already separated
    the field (2.56 to 4.74 across methods), so there is nothing to fix.

    Surface adherence is gone. The driving video itself scored 4.69 on it and the
    frozen generator beat us, so it was not supporting a claim.

ONE QUESTION PER CALL, ON PURPOSE
    Q3 carries a reference image that Q2 must not see: "rate stability only" is not
    a reliable instruction once a picture of the correct shape is sitting in the same
    context. Separate calls cost more and keep the two answers independent.
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

E = Path('/net/projects/ranalab/rajhansini/TRELLIS.2/experiments/dynamesh')
VIDS = Path('/net/projects/ranalab/rajhansini/baselines4d/_vids')
REFS = E / 'out/JUDGE/shape_refs'
ENV = E / '.env.judge'
METHODS = ['frozen', 'meshnca', 'l4gm', 'sv4d2', 'dg4d', 'ours']
VIEWS = ['train', 'diagA', 'diagB', 'diagC']
MODEL = 'gemini-3.6-flash'

Q2 = """This video shows a 3D object whose surface appearance changes over time.

How stable is the surface over time? Consider flicker, popping, and detail that
appears and disappears between frames.

5 = no flicker, popping or sudden jumps anywhere
4 = one or two faint shimmers
3 = visible flicker or popping in places
2 = frequent flicker; surface detail appears and disappears between frames
1 = severe flicker; the surface is unstable throughout

A video in which nothing changes at all is stable: score it 5.
Rate stability only, not whether the effect looks good or correct."""

Q3 = """The FIRST image shows the TRUE shape of the object, rendered untextured from
exactly the same camera as the video that follows.

The SECOND input is a video of one system's output for this object.

Is the object in the video the same shape as the reference? Judge the silhouette, the
proportions, and whether parts are missing, added, inflated or distorted.

5 = the same shape; outline and proportions match the reference
4 = essentially the same shape, with minor deviation
3 = recognisably this object, but noticeably distorted, inflated, or with parts
    missing or added
2 = substantially the wrong shape
1 = a different object

IGNORE ALL COLOUR AND TEXTURE. The video is textured and the reference is plain grey;
that difference is expected and must not affect your score. Judge shape only."""

QUESTIONS = {'temporal_coherence': (Q2, False), 'geometry_correctness': (Q3, True)}
SCHEMA = {'type': 'object',
          'properties': {'reason': {'type': 'string'}, 'score': {'type': 'integer'}},
          'required': ['reason', 'score']}
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


def vid_part(p, fps=10):
    return {'inline_data': {'mime_type': 'video/mp4',
                            'data': base64.b64encode(p.read_bytes()).decode()},
            'video_metadata': {'fps': fps}}


def img_part(p):
    return {'inline_data': {'mime_type': 'image/png',
                            'data': base64.b64encode(p.read_bytes()).decode()}}


def call(key, parts, prompt, timeout=300):
    body = {'contents': [{'parts': parts + [{'text': prompt}]}],
            'generationConfig': {'responseMimeType': 'application/json',
                                 'responseSchema': SCHEMA,
                                 'mediaResolution': 'MEDIA_RESOLUTION_LOW',
                                 'temperature': 1.0}}
    req = urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    out = json.loads(r['candidates'][0]['content']['parts'][0]['text'])
    return int(out['score']), out.get('reason', ''), r.get('usageMetadata', {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--objects', default='')
    p.add_argument('--all', action='store_true')
    p.add_argument('--questions', default='temporal_coherence,geometry_correctness')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    all_objs = sorted({f.name.split('__')[0] for f in VIDS.glob('*.mp4')})
    objs = all_objs if a.all else [o for o in a.objects.split(',') if o]
    bad = [o for o in objs if o not in all_objs]
    if bad:
        sys.exit(f'unknown objects: {bad}')
    qsel = [q for q in a.questions.split(',') if q]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r['obj'], r['method'], r['view'], r['question'], r['repeat']))
            except Exception:                                       # noqa: BLE001
                pass

    units = []
    missing_ref = set()
    for obj in objs:
        for rep in range(a.repeats):
            for m in METHODS + ['gt']:
                for v in (VIEWS if m != 'gt' else ['train']):
                    if not vpath(obj, m, v).exists():
                        continue
                    for q in qsel:
                        # Q3 without its reference would silently become a different
                        # question, so the cell is dropped and counted, never guessed.
                        if QUESTIONS[q][1] and not (REFS / f'{obj}__{v}.png').exists():
                            missing_ref.add(f'{obj}|{v}')
                            continue
                        if (obj, m, v, q, rep) not in done:
                            units.append((obj, m, v, q, rep))
    random.Random(0).shuffle(units)
    if missing_ref:
        log(f'WARNING: {len(missing_ref)} (obj,view) pairs have no shape reference; '
            f'their geometry cells are SKIPPED: {sorted(missing_ref)[:6]}')
    log(f'{len(done)} already done, {len(units)} to run, {a.workers} workers')

    key = load_key()
    state = {'n': 0, 'tok': 0}
    fh = out.open('a')

    def one(u):
        obj, m, v, q, rep = u
        prompt, needs_ref = QUESTIONS[q]
        parts = [img_part(REFS / f'{obj}__{v}.png')] if needs_ref else []
        parts.append(vid_part(vpath(obj, m, v)))
        for i in range(6):
            try:
                s, reason, usage = call(key, parts, prompt)
                rec = dict(obj=obj, method=m, view=v, question=q, repeat=rep,
                           score=s, reason=reason[:400],
                           tokens=usage.get('totalTokenCount'))
                with _lock:
                    fh.write(json.dumps(rec) + '\n'); fh.flush()
                    state['n'] += 1
                    state['tok'] += usage.get('totalTokenCount') or 0
                    if state['n'] % 50 == 0:
                        log(f"  {state['n']}/{len(units)} done, "
                            f"{state['tok']/1e6:.2f}M tokens")
                return
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and i < 5:
                    time.sleep(2 ** i + random.random() * 2); continue
                log(f'  FAIL {e.code} {obj}/{m}/{v}/{q}#{rep}: '
                    f'{e.read().decode()[:100]}')
                return
            except Exception as e:                                  # noqa: BLE001
                if i < 5:
                    time.sleep(2 ** i + random.random() * 2); continue
                log(f'  FAIL {obj}/{m}/{v}/{q}#{rep}: {type(e).__name__} {e}')
                return

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, units))
    fh.close()
    log(f'done {state["n"]}/{len(units)} in {(time.time()-t0)/60:.1f} min, '
        f'{state["tok"]/1e6:.2f}M tokens')


if __name__ == '__main__':
    main()
