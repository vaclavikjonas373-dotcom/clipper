"""context emphasis + sound effects: offline unit tests (no network, no key).

The detector model is never downloaded here — every test feeds hand-built
detections (the `index` dicts context.py caches) through the pure logic.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import clips as C      # noqa: E402
import context         # noqa: E402
import sfx             # noqa: E402


def words_for(pairs):
    """Word dicts from (word, start) pairs, 0.3 s each."""
    return [{"w": w, "s": s, "e": s + 0.3} for w, s in pairs]


def idx_from(rows):
    """A visual index dict: rows of (time, [(class, conf, cx, cy, w, h), ...])."""
    return {"times": [t for t, _ in rows], "dets": [d for _, d in rows]}


CAR = ["car", 0.8, 0.45, 0.5, 0.09, 0.07]          # near the face
FAR_CAR = ["car", 0.8, 0.9, 0.5, 0.09, 0.07]       # across the frame
DOG = ["dog", 0.7, 0.5, 0.6, 0.09, 0.07]


class Context(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        video = Path(cls.temp.name) / "ctx_test.mp4"
        subprocess.run([C.FFMPEG, "-y", "-v", "error",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
                        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=10",
                        "-shortest", "-pix_fmt", "yuv420p", str(video)], check=True)
        cls.video = video

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_mentions_map_to_classes(self):
        words = words_for([("The", 0.0), ("KFC", 0.4), ("car", 0.8),
                           ("is", 1.2), ("here", 1.6)])
        ms = context.mentions(words, 0, 10)
        by_word = {m[2]: m[3] for m in ms}
        self.assertIn("car", by_word["car"])                      # a plain class
        self.assertIn("pizza", by_word["kfc"])                    # the food group
        self.assertEqual(len(ms), 2, f"unmapped words leaked: {ms}")

    def test_find_scans_the_window(self):
        idx = idx_from([(10.0, [CAR]), (10.5, []), (11.0, [CAR]), (20.0, [CAR])])
        hits = context._find(idx, ("car",), 9.8, 11.5)
        self.assertEqual(len(hits), 2, hits)
        self.assertFalse(context._find(idx, ("dog",), 9.8, 11.5))

    def test_zoom_when_object_is_near_the_face(self):
        words = words_for([("Bro", 10.0), ("the", 10.4), ("car", 12.0)])
        idx = idx_from([(12.0, [CAR]), (12.5, [CAR]), (13.0, [CAR])])
        shots = [(0.0, 12.0, "crop", [(0.2, 60, 120), (0.8, 62, 122),
                                      (1.4, 64, 124), (2.0, 66, 126)]),
                 (12.0, 30.0, "crop", [(0.2, 80, 130), (1.0, 82, 132)])]
        clip = {"start": 10.0, "end": 40.0}
        ev = context.events(self.video, words, clip, idx, shots)
        self.assertEqual(len(ev), 1, ev)
        self.assertEqual(ev[0]["kind"], "zoom", ev)
        self.assertGreaterEqual(ev[0]["z"], 1.08)
        self.assertGreaterEqual(ev[0]["t0"], 1.0)                 # never at the very open
        self.assertLessEqual(ev[0]["t1"], 30.0 - 0.8)

    def test_focus_when_object_is_far_from_the_face(self):
        words = words_for([("Bro", 10.0), ("the", 10.4), ("car", 12.0)])
        idx = idx_from([(12.0, [FAR_CAR]), (12.5, [FAR_CAR]), (13.0, [FAR_CAR])])
        shots = [(0.0, 30.0, "crop", [(0.2, 60, 120), (1.0, 62, 122)])]
        ev = context.events(self.video, words, {"start": 10.0, "end": 40.0}, idx, shots)
        self.assertEqual(len(ev), 1, ev)
        self.assertEqual(ev[0]["kind"], "focus", ev)
        self.assertEqual(ev[0]["focus_pts"][0], (62, 122))        # reaction first

    def test_no_events_without_visibility_or_layout(self):
        words = words_for([("Bro", 10.0), ("the", 10.4), ("car", 12.0),
                           ("car", 20.0), ("again", 20.4)])
        shots = [(0.0, 30.0, "crop", [(0.2, 60, 120), (1.0, 62, 122)])]
        ev = context.events(self.video, words, {"start": 10.0, "end": 40.0},
                            idx_from([(12.0, [CAR])]), shots)     # one blurry sample
        self.assertEqual(ev, [], ev)
        ev = context.events(self.video, words, {"start": 10.0, "end": 40.0},
                            idx_from([(12.0, [CAR]), (12.5, [CAR]), (13.0, [CAR])]),
                            [(0.0, 30.0, "pad", [])])             # padded shot: play normally
        self.assertEqual(ev, [], ev)

    def test_dedup_gap_and_cap(self):
        # the same class twice counts once; a class with nothing on screen counts never
        words = words_for([("car", 11.0), ("car", 21.0), ("dog", 12.0),
                           ("dog", 22.0), ("clock", 14.0), ("book", 16.0)])
        idx = idx_from([(11.0, [CAR, DOG]), (11.5, [CAR]), (12.0, [CAR]), (12.5, [CAR]),
                        (21.0, [CAR]), (21.5, [CAR]),
                        (22.0, [DOG]), (22.5, [DOG]), (23.0, [DOG])])
        shots = [(0.0, 30.0, "crop", [(0.2, 60, 120), (1.0, 62, 122)])]
        ev = context.events(self.video, words, {"start": 10.0, "end": 40.0}, idx, shots)
        kinds = [e["cls"] for e in ev]
        self.assertEqual(len(kinds), len(set(kinds)), kinds)      # never twice per class
        self.assertLessEqual(len(ev), C.CTX_MAX_EVENTS, kinds)
        for a, b in zip(ev, ev[1:]):
            self.assertGreaterEqual(b["t0"], a["t1"] + C.CTX_GAP - 0.01, "events too close")


class Sfx(unittest.TestCase):
    EV = [{"t0": 2.0, "t1": 4.0, "kind": "zoom"},
          {"t0": 8.0, "t1": 9.5, "kind": "focus"}]

    def test_zoom_pair_and_focus_pop(self):
        out = sfx.plan(self.EV, 20, seed=5)
        kinds = [s["kind"] for s in out]
        self.assertEqual(kinds, ["whoosh", "thump", "pop"], out)
        self.assertAlmostEqual(out[0]["t"], 2.0, delta=0.05)
        self.assertAlmostEqual(out[1]["t"], 2.42, delta=0.05)
        self.assertTrue(all((s["path"]).exists() for s in out))

    def test_deterministic_and_varied(self):
        a, b = sfx.plan(self.EV, 20, seed=5), sfx.plan(self.EV, 20, seed=5)
        self.assertEqual(a, b, "same seed must give the same plan")
        c = sfx.plan(self.EV, 20, seed=6)
        self.assertNotEqual(a, c, "different seed should jitter differently")
        for s in a:
            self.assertTrue(0.92 <= s["rate"] <= 1.08, s)
            base = sfx.BASE[s["kind"]] * C.SFX_VOLUME
            self.assertTrue(base * 0.8 <= s["gain"] <= base * 1.2, s)

    def test_guards(self):
        early = sfx.plan([{"t0": 0.2, "t1": 1.0, "kind": "focus"}], 20, seed=1)
        late = sfx.plan([{"t0": 18.8, "t1": 19.5, "kind": "focus"}], 20, seed=1)
        self.assertEqual(early, [], "sounds at the very start")
        self.assertEqual(late, [], "sounds past the ending guard")
        many = sfx.plan([{"t0": t, "t1": t + 1, "kind": "focus"}
                         for t in (2, 4, 6, 8, 10)], 20, seed=1)
        self.assertLessEqual(len(many), C.SFX_MAX, "cap ignored")
        pair = sfx.plan([{"t0": 2.0, "t1": 2.6, "kind": "focus"},
                         {"t0": 2.5, "t1": 3.0, "kind": "zoom"}], 20, seed=1)
        gaps = [b["t"] - a["t"] for a, b in zip(pair, pair[1:])
                if a["ev"] != b["ev"]]
        self.assertTrue(all(g >= sfx.MIN_GAP for g in gaps), pair)

    def test_chain_graph(self):
        sounds = [{"kind": "pop", "t": 2.0, "rate": 1.0, "gain": 0.16,
                   "path": sfx.DIR / "pop.wav", "ev": 0},
                  {"kind": "whoosh", "t": 3.5, "rate": 0.95, "gain": 0.13,
                   "path": sfx.DIR / "whoosh.wav", "ev": 1}]
        g = sfx.chain(sounds, 20)
        self.assertIn("adelay=2000:all=1", g)
        self.assertIn("adelay=3500:all=1", g)
        self.assertIn("amix=inputs=3:normalize=0:duration=first", g)
        self.assertIn("loudnorm=I=-14:TP=-1.5:LRA=11", g)
        self.assertTrue(g.endswith("[aout]"))
        self.assertEqual(g.count("["), g.count("]"))


class WeaveAndZoom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        video = Path(cls.temp.name) / "wz_test.mp4"
        subprocess.run([C.FFMPEG, "-y", "-v", "error",
                        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
                        "-pix_fmt", "yuv420p", str(video)], check=True)
        cls.video = video

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_weave_focus_orders_object_first(self):
        shots = [(0.0, 10.0, "crop", [(0.2, 60, 120), (1.0, 62, 122)])]
        out = C.weave_focus(shots, [{"kind": "focus", "t0": 2.0, "t1": 3.5,
                                     "focus_pts": [(60, 120), (288, 120)]}])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1], (2.0, 3.5, "focus", [(288, 120), (60, 120)]))
        self.assertEqual(out[0], shots[0])

    def test_zoom_stage(self):
        self.assertEqual(C.zoom_stage([], self.video), "")
        ev = [{"kind": "zoom", "t0": 0.5, "t1": 1.5, "z": 1.3,
               "face": (0.4, 0.5), "kfs": [(0.95, 0.45, 0.5)]}]
        stage = C.zoom_stage(ev, self.video)
        self.assertIn("zoompan", stage)
        self.assertIn("fps=25", stage)
        self.assertIn("scale=1620:2880", stage)
        z, x, y = C.zoom_exprs(ev)
        for e in (z, x, y):
            self.assertIn("out_time", e)
            self.assertLess(len(e), 4000, e)
        self.assertIn("1.3000", z)
        self.assertIn("105.9", x)           # the clamped window offset toward the object


if __name__ == "__main__":
    unittest.main()
