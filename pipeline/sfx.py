#!/usr/bin/env python3
"""sfx.py — tiny original sound effects for the emphasis moves, made and mixed free.

Every sound is synthesized here with ffmpeg (pipeline/sfx/README.md has the
recipes), so the project owns them outright — nothing sampled, nothing licensed.
They are kept short and quiet, sit under the voice, never fire twice for the
same move, and get a small random rate/gain variation so repeats don't sound
copied. Turn them off with --no-sfx or {"SFX": false} in settings.local.json.
"""

import random
from pathlib import Path

import clips as C

DIR = Path(__file__).parent / "sfx"

LEN = {"pop": 0.14, "whoosh": 0.42, "thump": 0.28, "tick": 0.06}   # file lengths
# gain at volume 1. Calibrated on the sample source (loud trailer-music bed,
# bed peak ~-7 dBFS): at the default volume 0.7 each sound then peaks ~8-12 dB
# under the bed — an audible accent, still under the voice. Measured before the
# recalibration the mix moved by at most +1 dB (max volume) — inaudible.
BASE = {"pop": 0.40, "whoosh": 0.70, "thump": 0.45, "tick": 0.30}
MIN_GAP = 1.1        # two different moves never sound closer than this
EARLIEST = 0.5       # nothing before this into the clip (the hook opens clean)
LATE = 1.6           # nothing starts closer than this to the clip's end


def plan(events, dur, seed=0, cap=None, volume=None):
    """Where each sound goes, one list entry per sound: {kind, path, t, rate, gain}.

    A zoom move gets a whoosh as the punch-in starts and a soft thump as it
    settles; a focus reframe gets a pop. Times are seconds from clip start,
    jittered a little so the same move never lands identically twice. The seed
    (e.g. the clip's start time) keeps re-renders identical.
    """
    cap = C.SFX_MAX if cap is None else cap
    volume = C.SFX_VOLUME if volume is None else volume
    rng = random.Random(seed)
    units = []
    for k, e in enumerate(events):
        t0 = e.get("t0", 0.0)
        if t0 < EARLIEST or t0 > dur - LATE:
            continue                       # too early or too late to land well
        if e["kind"] == "zoom":
            units += [(t0, "whoosh", k), (t0 + 0.42, "thump", k)]
        elif e["kind"] == "focus":
            units.append((t0, "pop", k))
    units = [u for u in units if (DIR / f"{u[1]}.wav").exists()]
    out, last = [], None
    for t, kind, ev in sorted(units):
        if len(out) >= cap:
            break
        t = max(0.35, t + rng.uniform(-0.03, 0.03))   # a little human sloppiness
        if t + LEN[kind] > dur - 0.05:
            continue
        if out and ev != last and t - out[-1]["t"] < MIN_GAP:
            continue                       # one move at a time, sonically too
        out.append({"kind": kind, "ev": ev, "t": round(t, 3),
                    "path": DIR / f"{kind}.wav",
                    "rate": round(rng.uniform(0.92, 1.08), 3),
                    "gain": round(BASE[kind] * volume * rng.uniform(0.82, 1.18), 4)})
        last = ev
    return out


def chain(sounds, dur):
    """Audio filtergraph over the extra effect inputs, ending in [aout].

    Input 0 (the clip's own audio) must already be a filter_complex input;
    each sound is one extra -i. Everything is brought to 44.1 kHz stereo,
    delayed to its moment, gained, then mixed without level normalization
    (loudnorm after the mix evens the clip as a whole, as before).
    """
    parts = ["[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0]"]
    mix = ["[a0]"]
    for i, s in enumerate(sounds, 1):
        parts.append(
            f"[{i}:a]asetrate={int(44100 * s['rate'])},aresample=44100,"
            f"aformat=sample_rates=44100:channel_layouts=stereo,"
            f"adelay={int(s['t'] * 1000)}:all=1,volume={s['gain']:.4f}[a{i}]")
        mix.append(f"[a{i}]")
    parts.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=first[am]")
    parts.append(f"[am]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:st=0:d=0.12,"
                 f"afade=t=out:st={max(0, dur - 0.15):.2f}:d=0.15[aout]")
    return ";".join(parts)
