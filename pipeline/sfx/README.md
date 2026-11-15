# Sound effects (pop / whoosh / thump / tick)

Tiny sounds the pipeline mixes under the emphasis moves from `context.py`
(zoom punch-in → whoosh + soft thump, focus reframe → pop). They are
**synthesized here with ffmpeg** — no samples, no downloads, nothing to
license: the project owns them outright.

Placed by `sfx.plan()`, mixed by `sfx.chain()`; each playback gets a small
rate/gain variation so repeats don't sound copied. Disable with `--no-sfx`
or `{"SFX": false}` in `settings.local.json`.

## Regenerate

From the repo root:

```sh
ffmpeg -y -v error -f lavfi -i "aevalsrc=exprs='0.65*sin(2*PI*(650-2600*t)*t)*exp(-26*t)':s=44100:d=0.14,afade=t=out:st=0.1:d=0.04" -ac 2 -ar 44100 -c:a pcm_s16le pipeline/sfx/pop.wav
ffmpeg -y -v error -f lavfi -i "anoisesrc=color=pink:r=44100:d=0.42:a=0.85,bandpass=f=1100:width_type=o:w=1.6,afade=t=in:st=0:d=0.2,afade=t=out:st=0.2:d=0.22,volume=1.5" -ac 2 -ar 44100 -c:a pcm_s16le pipeline/sfx/whoosh.wav
ffmpeg -y -v error -f lavfi -i "aevalsrc=exprs='0.85*sin(2*PI*52*t)*exp(-14*t)+0.3*sin(2*PI*140*t)*exp(-24*t)':s=44100:d=0.28,afade=t=out:st=0.22:d=0.06" -ac 2 -ar 44100 -c:a pcm_s16le pipeline/sfx/thump.wav
ffmpeg -y -v error -f lavfi -i "aevalsrc=exprs='0.5*sin(2*PI*1900*t)*exp(-55*t)':s=44100:d=0.06,afade=t=out:st=0.04:d=0.02" -ac 2 -ar 44100 -c:a pcm_s16le pipeline/sfx/tick.wav
```

- **pop** — a fast falling chirp (650 → ~290 Hz) with a hard exponential decay.
- **whoosh** — band-passed pink noise that swells in and back out (~0.42 s).
- **thump** — a low 52 Hz sine decaying over ~0.28 s, with a faint 140 Hz tap.
- **tick** — a 1.9 kHz blip, 60 ms (kept for future transition sounds).

The lengths must match `sfx.LEN` — change both together. Keep them short and
quiet; they sit under the voice, never on top of it. How loud they sit in the
mix is calibrated in `sfx.BASE` (measured: each sound peaks ~8–12 dB under a
loud music bed at the default `SFX_VOLUME`).
