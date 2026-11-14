"""--no-ai local picks: synthetic transcript through pick_hooks_local (no network, no key)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import clips as C  # noqa: E402
import score  # noqa: E402


def words_for(sents, wps=0.35, gap=0.25):
    """Word dicts with timings for a list of sentence strings."""
    out, t = [], 0.0
    for s in sents:
        for w in s.split():
            out.append({"w": w, "s": t, "e": t + wps})
            t += wps
        t += gap
    return C.mark_breaks(out)


class NoAiPicks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        video = Path(cls.temp.name) / "noai_test.mp4"
        subprocess.run([C.FFMPEG, "-y", "-v", "error",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=320",
                        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=320",
                        "-shortest", "-pix_fmt", "yuv420p", str(video)], check=True)
        cls.video = video

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_picks_pass_editorial_rules(self):
        sents = []
        for k in range(80):
            if k == 6:
                sents.append("Bro oh my god that was so crazy! This is insane!")
            elif k % 3 == 0:
                sents.append("This is a normal sentence about the game we all play together.")
            elif k % 3 == 1:
                sents.append("The stream went completely wild after that crazy play!")
            else:
                sents.append("And then we kept on playing for another few minutes.")
        words = words_for(sents, 0.3, 0.2)
        total = max(w["e"] for w in words) + 1
        picks = C.pick_hooks_local(self.video, words, total, "gaming")
        self.assertGreaterEqual(len(picks), 3, f"only {len(picks)} picks survived")
        for p in picks:
            self.assertGreaterEqual(p["end"] - p["start"], C.MIN_S - 1)
            self.assertLessEqual(p["end"] - p["start"], C.MAX_S)
            first = next(w for w in words if w["s"] >= p["start"])
            self.assertFalse(C._bad_start(first), f"opens on filler: {first['w']}")
            self.assertTrue(C._capital(first), f"not sentence case: {first['w']}")
            self.assertTrue(p["title"], "no title on a local pick")
            self.assertEqual(len(p["hashtags"]), 3, f"tags: {p['hashtags']}")
            self.assertIn("#", " ".join(p["hashtags"]))
        for i, a in enumerate(picks):                      # strongest first, no overlap
            for b in picks[i + 1:]:
                self.assertLess(min(a["end"], b["end"]) + C.GAP, max(a["start"], b["start"]),
                                "picks overlap")
        # flat test audio leaves ranking to text signals; the crazy moment must survive
        kept = [" ".join(w["w"] for w in words if p["start"] <= w["s"] < p["end"]).lower()
                for p in picks]
        self.assertTrue(any("oh my god" in t for t in kept),
                        "hype moment cut away entirely")

    def test_category_guess(self):
        text = "Let's play the game bro this stream is crazy chat"
        self.assertEqual(score._guess_category(text), "gaming")
        self.assertEqual(score._guess_category("welcome to the podcast episode"), "podcast")
        self.assertEqual(score._guess_category("nothing matches here"), "talk")


if __name__ == "__main__":
    unittest.main()
