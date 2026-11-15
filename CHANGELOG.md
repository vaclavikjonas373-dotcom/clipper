# Changelog

Newest first. Every change to code, settings, rules or docs gets a dated entry here
(see `AGENTS.md`).

## 2026-11-15 — context emphasis (words ↔ picture) + sound effects
- New `pipeline/context.py`: spoken keywords are matched to what is actually on
  screen before the clip emphasizes anything. Each keyword maps to COCO
  detector classes ("KFC" → the food group; persons deliberately never a
  trigger); a **YOLOX-S** detector (OpenCV zoo, Apache-2.0, ~34 MB, downloaded
  once into `pipeline/models/`) scans the source at 2 fps, cached as
  `<video>.visualindex.json` beside the transcript. A mention counts only when
  the class is detected near the moment it is spoken (conf ≥ 0.40, sane box
  size). No verified match → the clip plays normally; nothing invented.
- Emphasis moves in `clips` mode, max 3 per clip, ≥ 2.5 s apart, inside one
  camera shot and away from the clip's edges: **zoom** — an eased punch-in on
  the finished frame (zoompan, supersampled 1.5×, clamped so the tracked face
  stays in frame) when the object is near it; **focus** — a split-style reframe
  (object on top, reaction below) woven in as a new shot mode, reusing the
  mode-switch machinery. Framing chains (crop/split/pad) unchanged.
- New `pipeline/sfx.py` + `pipeline/sfx/` (pop, whoosh, thump, tick):
  synthesized with ffmpeg, project-owned, nothing sampled. Zoom → whoosh into
  a soft thump; focus → pop. Rate/gain jitter, near-edge guards, per-clip cap,
  mixed under the voice before the existing loudnorm.
- `score.rank()` gained a free **match** signal (share of the hook's keywords
  verifiably on screen), weighted 0.5 in talk-family, gaming, sports, music.
- Flags `--no-context` / `--no-sfx` and settings `CONTEXT`, `CTX_MAX_EVENTS`,
  `CTX_GAP`, `SFX`, `SFX_VOLUME`, `SFX_MAX` (all honored from
  `settings.local.json`; flags beat settings). With both off, rendering is
  byte-path identical to before (plain `-af` audio path preserved).
- `clips.plan_shots()` face samples now carry (t, x, y); `fps_of()`/`has_audio()`
  helpers; `--check` gained an sfx-assets line and a soft context-detector line.
- Tests: `tests/test_context_sfx.py` (offline, no network, no model) — keyword
  mapping, visibility rules, zoom/focus geometry, dedup/gap/cap, sfx planning
  and filtergraphs, focus weaving and zoompan stages.
- Stdout is now line-buffered when piped (progress prints appeared stale in
  redirected logs on Windows because of block buffering).
- The report gained a "Context moves" table (per clip: move, where in the
  source, how long it is held) and `report.md` omits it when a run had none.
- Bugfix: the visual index was cached in `output/` instead of the per-video
  folder. Submodules do `import clips as C`, but the pipeline runs as a script,
  so Python executed `clips.py` a second time for them — a copy whose `OUT`
  was still the `output/` root default (`reedit.py`/`check_captions.py` already
  worked around this by rebinding `C.OUT` themselves). `clips.py` now
  registers its running copy under both module names, so `context`, `score`,
  `sfx` and `reedit` see the per-video `OUT`. The stray cache file was moved
  into `output/<video>/` (not deleted) and a re-run cache-hit on it there.
- Sound-effect gains recalibrated from an A/B on the sample source: at the old
  base gains the mix changed by at most +1 dB (max volume) — inaudible. At the
  new base gains the pop moves the mix by +3.0 dB (max) and the thump +2.2 dB
  over the same windows — audible accents still ~8–12 dB under a loud bed.
- Validated the emphasis render end-to-end on real footage (scratch, in
  `pipeline/dbg_render/`, labeled DBG): a real 28 s range of the sample source
  with real detector hits and clearly-synthetic word timings produced a zoom
  (1.35×, face kept in frame) and a focus reframe (car on top, reaction below),
  both correct in grabbed frames, with the sounds present. The sample's real
  transcript only ever names food words and no food is ever on screen (GTA
  trailer + webcam), so its five deliverable clips correctly play with no
  verified moves — the "nothing invented" rule working as designed.
- Updated README (feature, flags, alt text), `docs/HOW-IT-WORKS.md` (pipeline,
  score table, framing table, new sections, settings, limits), the skill
  (2.6.0, context rules, brief line), and the architecture diagrams
  (`render-system.py`/`system.svg`, `pipeline.d2`/`pipeline.svg`).
- Performance: the visual-index scan, `plan_shots()` and `score.picture()`
  decode in one ordered pass instead of seeking to every sample — on the
  long-GOP sample source a seek cost ~1.35 s against ~10 ms a sequential frame
  (measured), which is why the first full run stalled with no output.
  `plan_shots` validated against a strict decode ground truth (new code
  matches it exactly on all 41 samples of a 20 s range; the old seek version
  flapped off it on 23 — the rework is also more accurate). `score.picture()`
  keeps the ~0.2 s motion grid (diffs ≥ 0.18 s apart, never consecutive
  frames) and now samples faces every 0.75 s, capped at 4. Picture results are
  cached per hook window and camera-cut scans per range. Measured on a 20 s
  range: plan_shots 62.7 → 18.8 s, picture 47.9 → 8.9 s; a full index scan of
  the 3.7-min source went from ~10 min of seeking to ~3.5 min of decoding
  (cached afterwards as `<video>.visualindex.json`).

## 2026-11-14 — free offline moment picking (`--no-ai`)
- Added `pick_hooks_local()` to `pipeline/clips.py`: with `--no-ai`, `clips` mode
  picks moments without any API. Sentence-start candidates, one to four sentence
  closes as their end (short to long inside the length window, so mid-length
  picks don't block better short ones), ranked by excited language (hype
  phrases, exclamations, laughter, numbers — opening sentence weighted highest)
  plus loudness peak/energy, then run through the same `_validate` checks and
  free local signals in `score.rank(use_ai=False)`. Titles, hashtags and the
  category are derived locally (`score._guess_category()` or `--category=`).
- `--no-ai` previously worked only in `visual` mode; `reedit` still needs the AI
  (clearer error message). `chat()` returns `None` offline so no path touches
  the network or key; report notes "offline picks".
- Fixed face detection failing on machines whose user path has non-ASCII
  characters (e.g. "Jonáš"): OpenCV's model reader is not Unicode-aware on
  Windows, so `pipeline/models/yunet.onnx` "couldn't be read" from an absolute
  path. New `clips.cv_path()` hands OpenCV a relative or short ASCII path;
  `--check` now really loads the detector after opening a capture, so this
  class of failure can't pass silently again.
- Added `tests/test_noai_picks.py` (offline, synthetic transcript) covering the
  editorial rules and the category guess.
- Updated README, `docs/HOW-IT-WORKS.md`, the skill, the system/pipeline
  diagrams (`render-system.py`, `pipeline.d2`, overview footer) and their
  rendered files.

## 2026-10-23 — README architecture research layer
- Updated the README's main system graphic and alt text to show agent research,
  optional scraping, campaign fit/permission checks and local context before the
  existing video engine. Rendering behavior is unchanged.
- Made main-graphic regeneration and visual inspection explicit in agent/skill
  rules; diagram freshness alone does not prove architectural completeness.

## 2026-10-23 — campaign research and maintained documentation
- Documented agent-assisted discovery, extraction, campaign/footage fit checks,
  evidence status and optional Firecrawl use without adding an API integration.
- Added task-triggered freshness and documentation maintenance rules; no scheduler,
  paid tool, credential or pipeline behavior was changed.
- Kept personalized research and account context local; updated the collaboration
  diagram to distinguish research from the rendering pipeline.

## 2026-10-23 — campaign asset organization
- Added a shared procedure for organizing supplied briefs and permitted media,
  verifying moves and tracking source rights, missing inputs and output paths.
- Campaign source folders and planning/output workspaces use existing ignored
  `sources/` and `output/`; explicit video paths preserve pipeline compatibility.

## 2026-10-22 — portable continuity and runtime repair
- All agents discover optional private `CONTEXT.local.md`; private context stays ignored.
- `CLAUDE.md` imports `AGENTS.md` so Claude Code loads the same project rules.
- Reinstalling Hermes retargets an existing profile's working folder, backs up
  replaced configuration/links and preserves model/provider settings.
- Fixed `CDPATH` causing duplicated working-folder values; a different checkout
  now requires explicit `--retarget`, preventing accidental staging-folder use.
- Added local `FFMPEG_PATH` support, without committing machine-specific paths.
- Isolated unchanged Whisper inference in a worker to avoid PyAV/OpenCV macOS
  library collisions; dependency checks import each package in its own process.
- Consolidated campaign settings and Git instructions in the skill; documented
  current payment/platform research and the limits of the hook score.
- Reruns preserve previous clips, post text, reports and review sheets in
  `history/run-*/` instead of deleting them.

## 2026-10-21 — README component architecture
- Replaced the README's simplified illustration with a detailed system diagram:
  local components, three mode paths, cache, shared rendering, AI boundary and outputs.
- Preserved the illustration on `docs/VISUAL-OVERVIEW.md`.
- Added a dependency-free SVG renderer and included its output in the existing
  diagram freshness check. Verified paths and conditional outputs against the code.

## 2026-10-21 — illustrated README overview
- Replaced the README's text-heavy diagram with a custom four-step illustration:
  source video, timed speech, three selection modes and captioned phone previews.
- Kept detailed D2 diagrams in the architecture guide, linked from the overview.
- Added a local Pillow renderer, bundled licensed fonts and design notes so the
  graphic can be updated without a paid graphics service.

## 2026-10-21 — portrait architecture refresh
- Redesigned both D2 diagrams as numbered portrait flows with a restrained palette,
  consistent cards and no crossing connectors. Updated the README image presentation.
- Clarified that re-edits skip clip ranking and produce a separate set of outputs;
  kept the diagrams and architecture guide aligned with the existing code.

## 2026-10-21 — architecture diagrams
- `docs/ARCHITECTURE.md` with two D2 diagrams (pipeline; people, agents and GitHub flow),
  shown in the README. Sources in `docs/architecture/*.d2`, rendered by `render.sh`
  (D2 v0.9.0). CI fails if a diagram image is out of date.

## 2026-10-22 — prepared for public release
- Added the MIT `LICENSE`; README licence and disclaimer section.
- Removed the committed `trends/` example (a real creator's measured Shorts); `trends/` is now git-ignored, since briefs are generated per user.
- Removed a machine-specific ffmpeg path and personal log references; examples use placeholder channels.
- `AGENTS.md`: agents on forks or without push access keep changes local instead of opening pull requests.
- Commit author emails rewritten to the GitHub no-reply address.
- README: Contributors section (@marcxxv as maintainer; @blame-zeus as contributor; a conversation about his clipping work inspired the project).

## 2026-10-21 — hook scoring, trends, automatic git workflow
- `score.py`: every candidate's first 3 s scored on voice (energy, pitch variation, pace,
  dead air), picture (face, motion, cuts) and an AI look at the phone-cropped frames
  (NYO glm-5.3-flash, which accepts images), weighted by detected content category.
  Clips numbered best-first; scores and strengths in the report. `--category=` override.
- `trends.py`: measures the most-viewed Shorts of channels (`@name`) or hashtags; writes
  `trends/<name>.md`; `clips.py --trend=<name>` adds that brief to the AI's instructions.
- `settings.local.json` (git-ignored) for personal values such as campaign lengths.
- `--no-ai` offline mode and `CLIPS_WHISPER` model override; `ROUNDS` 2 → 3.
- GitHub: `main` protected (pull requests required, owner bypasses); `checks` workflow
  installs and runs an offline clip on Windows and Linux for every push and PR.
- `AGENTS.md`: owner's agent pushes automatically; a collaborator's agent opens pull
  requests by itself for tested, general improvements.
- Windows setup falls back to Chocolatey when winget is missing.

## 2026-10-21 — published as a repository
- Repository layout: one skill at `.agents/skills/clipping/` (read by OpenCode, Hermes,
  Claude Code, Codex and the pipeline itself); setup scripts for Windows and macOS/Linux;
  OpenCode and Hermes integrations; docs split into SETUP, OPENCODE, HERMES, HOW-IT-WORKS.
- Windows support: UTF-8 file handling, relative caption paths (a drive letter breaks
  ffmpeg filters), ffmpeg/ffprobe found on PATH, downloads robust to non-English titles.
- `clips.py --check` self-test (ffmpeg, fonts, face model, packages, NYO key and model).
- NYO key read from `NYO_API_KEY` or the repo's `.env` as well as a Hermes profile.
- Only the skill's editorial brief (between `brief` markers) is sent to the AI: about
  1,100 characters instead of 3,000 per request.

## 2026-10-21 — posting helpers
- A `.txt` beside every clip and re-edit with title and hashtags, ready to paste.
- Optional `--headline`: AI hook text at the top of normal clips (off by default).

## 2026-10-21 — captions and re-edits
- Karaoke captions: the spoken word eases to amber and 120% as it is said; Montserrat Black.
- Word timings snapped to the audio: start error 119 → 21 ms median (ground-truth test);
  no caption during pauses (was shown in 60% of pauses).
- Re-edits: 0.2 s cross-fades of picture and sound; nearby lines play through; punch-in
  only for same-shot jumps; progress bar removed.
- Fixed: re-edit picture stopping early (one-frame offset). Every file is now checked for
  picture/sound length.
- Japanese/Chinese: phrase breaks as sentence ends, never open on a particle; hyphenated
  words re-joined.

## 2026-10-20 — modes for every video type
- Auto mode: `clips` (talk), `reedit` (trailer), `visual` (no speech); `--mode=` override.
- Per-shot framing: face crop, two-speaker split, padded wide shots.
- Clips start and end on real sentences; overlap and ranking fixed; second AI round.
- Language-aware text, long videos read in 40-minute windows, YouTube link input.
