"""llm_judge.py -- semantic evaluation of the 3DV results with Gemini as the judge.

WHY THIS EXISTS
    Flicker, acceleration and drift are pixel/texel statistics. They say nothing
    about whether the video shows the *intended effect*, or whether the texture is
    bonded to the surface rather than swimming over it. This scores those.

PROTOCOL (Itai, 2026-08-30)
    Per result, render the training view plus 3 unseen views, hand each video to an
    LLM, ask it to rate a question. Repeat and average, because it is noisy.
    Do it for every compared method -> a comparative quantitative table.

    Everything is already rendered: baselines4d/_vids/ holds 1050 mp4s,
      <obj>__<method>.mp4            train view
      <obj>__<view>__<method>.mp4    view in {diagA,diagB,diagC}
    42 objects x 4 views x 6 methods, plus gt at the training view. No holes.

TWO CALL TYPES
    A. effect_match  -- train view only, GT driving video + the method video, both
       sent in one request. We do NOT hand the judge a text description of the
       effect: the Kling prompts are 400-word negation blocks and hand-writing 42
       effect strings would leak our own wording into the score. The GT video IS
       the intended effect. gt-vs-gt is submitted too and must come back 5; it is
       the calibration check, not a result row.
    B. per_video     -- all 4 views, one video, no reference. Temporal smoothness,
       geometry preservation and surface adherence are all judgeable from a single
       clip without knowing what the effect was supposed to be.

THE ONE SETTING THAT DECIDES WHETHER THIS WORKS AT ALL
    Gemini samples video at 1 FPS by default. A 6-second clip would reach the judge
    as SIX frames, and asking six frames about flicker is asking nothing. We set
    videoMetadata.fps explicitly. --fps 10 over 6 s = 60 frames, which resolves
    frame-to-frame popping while staying affordable at low media resolution.
    Token usage is recorded per call so the pilot can price the full run.

BLINDING
    Only bytes are sent, never filenames, and the method is never named in the
    prompt. Work order is shuffled so a rate-limit stall cannot correlate with a
    method.

Usage
    python3 jobs/llm_judge.py --objects spot_lava,chair_moss --repeats 3 \
        --out out/JUDGE/pilot.jsonl 2>&1 | tee logs/judge_pilot.log
    python3 jobs/llm_judge.py --all --repeats 3 --out out/JUDGE/full.jsonl
Resume is automatic: completed (video,question,repeat) keys in --out are skipped.
"""
import argparse
import base64
import json
import os
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
# gt is the driving video, not a method: it is the reference for A and the topline
# row for B. It exists at the training view only.
METHODS = ['frozen', 'meshnca', 'l4gm', 'sv4d2', 'dg4d', 'ours']
VIEWS = ['train', 'diagA', 'diagB', 'diagC']

SCALE = """Rate on an integer scale from 1 to 5. Use the whole scale; 3 means
genuinely middling, not "unsure". Judge only what the question asks about and
ignore image resolution, framing and rendering style."""

QUESTIONS = {
    'effect_match': dict(kind='paired', text="""The FIRST video is the reference: it shows the intended surface effect.
The SECOND video shows another system's attempt to reproduce that same effect on a 3D model.

How well does the second video reproduce the intended effect of the first?

5 = the same effect, matching in appearance, extent and how it progresses over time
4 = clearly the same effect, with minor differences in look, extent or timing
3 = a recognisably related effect, but noticeably different in appearance or extent
2 = only loosely related; the general idea is present but the appearance is wrong
1 = a different effect entirely, or no effect at all

Judge the effect only. The two videos may differ in object pose, resolution and
playback speed; none of that should change your score."""),

    'temporal_smoothness': dict(kind='single', text="""This video shows a 3D object whose surface appearance changes over time.

How temporally smooth is it? Consider flicker, popping, and detail that appears and
disappears between frames.

5 = perfectly smooth; no flicker, popping or sudden jumps
4 = mostly smooth; occasional faint shimmer
3 = visible flicker or popping in places
2 = frequent flicker; surface detail appears and disappears between frames
1 = severe flicker; the surface is unstable throughout

A video in which nothing changes at all is smooth: score it 5 on this question.
Rate smoothness only, not whether the effect is good."""),

    'geometry_preservation': dict(kind='single', text="""This video shows a 3D object whose surface appearance changes over time.

How well is the object's own shape preserved throughout? The surface may change
freely; the underlying form should not.

5 = the shape is identical throughout; only the surface changes
4 = the shape is essentially stable, with very slight wobble
3 = noticeable deformation, drift or swelling of the shape
2 = the shape changes substantially over the video
1 = the shape is continuously morphing or becomes unrecognisable

Rate shape stability only, not the surface appearance."""),

    'surface_adherence': dict(kind='single', text="""This video shows a 3D object whose surface appearance changes over time.

Does the pattern behave like it is bonded to the surface, or like it is sliding over
it? A bonded pattern stays anchored where it started and grows or changes in place;
an unbonded one scrolls, swims or drifts across the form independently of it.

5 = firmly bonded; the pattern changes in place and stays anchored to the surface
4 = mostly bonded, with slight sliding
3 = noticeable sliding or swimming of the pattern across the surface
2 = the pattern drifts across the form largely independently of it
1 = the pattern is fully detached from the surface

Rate adherence only, not the quality or smoothness of the effect."""),
}

SCHEMA = {'type': 'object',
          'properties': {'reason': {'type': 'string'}, 'score': {'type': 'integer'}},
          'required': ['reason', 'score']}

_print_lock = threading.Lock()
_write_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def load_key():
    for line in ENV.read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):
            return line.split('=', 1)[1].strip()
    sys.exit(f'no GEMINI_API_KEY in {ENV}')


def vpath(obj, method, view):
    return VIDS / (f'{obj}__{method}.mp4' if view == 'train'
                   else f'{obj}__{view}__{method}.mp4')


def part(path, fps):
    return {'inline_data': {'mime_type': 'video/mp4',
                            'data': base64.b64encode(path.read_bytes()).decode()},
            'video_metadata': {'fps': fps}}


def call(key, model, parts, prompt, media_res, timeout=300):
    """One generateContent call. Returns (score, reason, tokens)."""
    body = {'contents': [{'parts': parts + [{'text': prompt + '\n\n' + SCALE}]}],
            'generationConfig': {'responseMimeType': 'application/json',
                                 'responseSchema': SCHEMA,
                                 'mediaResolution': media_res,
                                 'temperature': 1.0}}
    req = urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    out = json.loads(r['candidates'][0]['content']['parts'][0]['text'])
    return int(out['score']), out.get('reason', ''), r.get('usageMetadata', {})


def work(args, key, done, fh):
    """Build the shuffled unit list: one unit = (obj, method, view, question, rep)."""
    units = []
    for obj in args.objects:
        for rep in range(args.repeats):
            # A: effect match, train view, against GT. gt-vs-gt included as calibration.
            for m in METHODS + ['gt']:
                if vpath(obj, m, 'train').exists() and vpath(obj, 'gt', 'train').exists():
                    units.append((obj, m, 'train', 'effect_match', rep))
            # B: single-video questions, every view. gt only has the train view.
            for m in METHODS + ['gt']:
                for v in (VIEWS if m != 'gt' else ['train']):
                    if not vpath(obj, m, v).exists():
                        continue
                    for q in ('temporal_smoothness', 'geometry_preservation',
                              'surface_adherence'):
                        units.append((obj, m, v, q, rep))
    units = [u for u in units if key_of(u) not in done]
    random.Random(0).shuffle(units)
    return units


def key_of(u):
    return '|'.join(map(str, u))


def run_one(u, key, args, fh, state):
    obj, method, view, q, rep = u
    spec = QUESTIONS[q]
    parts = ([part(vpath(obj, 'gt', 'train'), args.fps)] if spec['kind'] == 'paired' else [])
    parts.append(part(vpath(obj, method, view), args.fps))
    for attempt in range(6):
        try:
            score, reason, usage = call(key, args.model, parts, spec['text'],
                                        args.media_res)
            rec = dict(obj=obj, method=method, view=view, question=q, repeat=rep,
                       score=score, reason=reason[:400],
                       tokens=usage.get('totalTokenCount'))
            with _write_lock:
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                state['n'] += 1
                state['tok'] += usage.get('totalTokenCount') or 0
                if state['n'] % 20 == 0:
                    log(f"  {state['n']}/{state['total']} done, "
                        f"{state['tok']/1e6:.2f}M tokens")
            return
        except urllib.error.HTTPError as e:
            code = e.code
            detail = e.read().decode()[:160]
            if code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt + random.random() * 2)
                continue
            log(f'  FAIL {code} {obj}/{method}/{view}/{q}#{rep}: {detail}')
            return
        except Exception as e:                                  # noqa: BLE001
            if attempt < 5:
                time.sleep(2 ** attempt + random.random() * 2)
                continue
            log(f'  FAIL {obj}/{method}/{view}/{q}#{rep}: {type(e).__name__} {e}')
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--objects', default='')
    p.add_argument('--all', action='store_true')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--model', default='gemini-3.6-flash')
    p.add_argument('--fps', type=int, default=10,
                   help='video sampling rate handed to the judge (default 1 is unusable)')
    p.add_argument('--media-res', default='MEDIA_RESOLUTION_LOW')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    all_objs = sorted({f.name.split('__')[0] for f in VIDS.glob('*.mp4')})
    args.objects = all_objs if args.all else [o for o in args.objects.split(',') if o]
    missing = [o for o in args.objects if o not in all_objs]
    if missing:
        sys.exit(f'unknown objects: {missing}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add(key_of((r['obj'], r['method'], r['view'],
                                 r['question'], r['repeat'])))
            except Exception:                                    # noqa: BLE001
                pass

    key = load_key()
    with out.open('a') as fh:
        units = work(args, key, done, fh)
        state = {'n': 0, 'tok': 0, 'total': len(units)}
        log(f'objects={len(args.objects)} repeats={args.repeats} '
            f'model={args.model} fps={args.fps} res={args.media_res}')
        log(f'{len(done)} already done, {len(units)} to run, '
            f'{args.workers} workers')
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(lambda u: run_one(u, key, args, fh, state), units))
        dt = time.time() - t0
    log(f'done {state["n"]}/{len(units)} in {dt/60:.1f} min, '
        f'{state["tok"]/1e6:.2f}M tokens')
    if state['n']:
        log(f'  {state["tok"]/state["n"]:.0f} tokens/call, '
            f'{dt/state["n"]:.1f} s/call wall')


if __name__ == '__main__':
    main()
