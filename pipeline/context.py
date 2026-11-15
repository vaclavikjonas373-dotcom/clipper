#!/usr/bin/env python3
"""context.py — match what is SAID to what is SHOWN, then emphasize it.

Free and entirely local: transcript keywords are mapped to everyday object
classes, a COCO object detector (YOLOX-S, Apache-2.0, downloaded once) scans
the source once per video (cached next to the transcript), and a mention only
counts when that thing is actually on screen around the moment it is spoken.
Verified matches become emphasis moves for the render:

  zoom   — the object is near the tracked face: a smooth punch-in toward it,
           clamped so the face stays in frame (zoompan, ~1.2-1.35x, eased)
  focus  — the object is elsewhere in the frame: a split-style reframe with
           the object in the top half and the reaction in the bottom half

Nothing is invented. When a word has no verified match the clip plays
normally; a generic detector can't verify brand logos, so "KFC" emphasizes
only real food/dining objects that are visible — never a faked logo.
Persons are never a trigger: people are everywhere in reaction videos, and
"he said a guy" is not a reason to move the camera.
"""

import json, re, time

import numpy as np

import clips as C

MODEL_NAME = "object_detection_yolox.onnx"
_URL = ("https://huggingface.co/opencv/object_detection_yolox/resolve/main/"
        "object_detection_yolox_2022nov.onnx")

CONF      = 0.40        # detection confidence that counts as "visible"
MIN_BOX   = 0.004       # skip specks: smallest box area (share of frame) to use
LOOK_BACK = 1.3         # a match may appear this long before the word
LOOK_AHEAD = 1.6        # ... or this long after
MAX_HOLD  = 2.8         # emphasis never lingers longer than this
MIN_ZOOM  = 1.08        # below this a punch-in is pointless → reframe instead
MAX_EVENTS = 3          # per clip (clips.CTX_MAX_EVENTS may override)

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# what the words can look like. "food" is a group: any of those classes counts.
_GROUPS = {"food": ("hot dog", "pizza", "donut", "cake", "sandwich", "bowl",
                    "dining table", "refrigerator", "cup", "bottle")}
_TRIGGERS = (
    ("car",         "car cars supercar racecar convertible sports car"),
    ("truck",       "truck trucks pickup"),
    ("bus",         "bus buses"),
    ("motorcycle",  "motorcycle motorcycles motorbike motorbikes"),
    ("airplane",    "plane planes airplane airplanes jet jets"),
    ("boat",        "boat boats ship ships yacht yachts"),
    ("train",       "train trains"),
    ("bicycle",     "bicycle bicycles"),
    ("bottle",      "bottle bottles"),
    ("cup",         "cup cups"),
    ("cell phone",  "phone phones"),
    ("laptop",      "laptop laptops computer computers"),
    ("clock",       "clock clocks"),
    ("book",        "book books"),
    ("dog",         "dog dogs puppy doggy"),
    ("cat",         "cat cats kitty"),
    ("horse",       "horse horses"),
    ("bird",        "bird birds"),
    ("sports ball", "ball balls football basketball soccer"),
    ("skateboard",  "skateboard skateboards"),
    ("surfboard",   "surfboard surfboards"),
    ("umbrella",    "umbrella umbrellas"),
    ("backpack",    "backpack backpacks"),
    ("suitcase",    "suitcase suitcases luggage"),
    ("teddy bear",  "teddy"),
    ("couch",       "couch couches sofa sofas"),
    ("chair",       "chair chairs"),
    ("tv",          "tv tvs television television"),
    ("food",        "kfc food eat eats eating hungry meal burger burgers "
                    "cheeseburger chicken snack pizza donut doughnut cake "
                    "sandwich taco tacos fries dinner breakfast"),
)
_WORDS, _PHRASES = {}, {}
for _cls, _ws in _TRIGGERS:
    _classes = _GROUPS[_cls] if _cls in _GROUPS else (_cls,)
    for _w in _ws.split():
        (_PHRASES if " " in _w else _WORDS)[_w] = _classes


# ---------------- model ----------------

def ensure_model():
    """The detector, downloaded once into pipeline/models (Apache-2.0)."""
    p = C.MODELS_DIR / MODEL_NAME
    if p.exists() and p.stat().st_size > 10_000_000:
        return True
    try:
        import requests
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   context: downloading the object detector (~34 MB, once)")
        tmp = p.with_suffix(".part")
        with requests.get(_URL, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(p)
        return True
    except Exception as e:
        print(f"   context: can't fetch the detector ({e}) — clips play normally")
        return False


_NET = None

def _net():
    """The loaded detector: (net, cv2, grids, strides). CPU is ~0.1 s a frame."""
    global _NET
    if _NET is None:
        import cv2
        net = cv2.dnn.readNet(C.cv_path(C.MODELS_DIR / MODEL_NAME))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        grids, strides = [], []
        for st in (8, 16, 32):
            xv, yv = np.meshgrid(np.arange(640 // st), np.arange(640 // st))
            g = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(g)
            strides.append(np.full((1, g.shape[1], 1), st))
        _NET = (net, cv2, np.concatenate(grids, 1), np.concatenate(strides, 1))
    return _NET


# ---------------- visual index ----------------

def index(video, total, step=0.5):
    """What is on screen when: one detector pass over the source, cached.

    Returns {"times": [...], "dets": [[name, conf, cx, cy, w, h], ...]}
    (positions as shares of the frame), or None when unavailable.
    """
    if total < 5:
        return None
    meta = {"model": "yolox-s-2022nov", "step": step, "total": round(total, 2)}
    cache = C.OUT / f"{video.stem}.visualindex.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("meta") == meta and d.get("times"):
                print(f"   context: cached visual scan ({len(d['times'])} frames)")
                return d
        except ValueError:
            pass
    if not ensure_model():
        return None
    net, cv2, grids, strides = _net()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    times, dets = [], []
    next_t, t0, said = 0.0, time.time(), time.time()
    while next_t < total - 0.05:
        ok, fr = cap.read()          # read in order: seeking to each sample
        if not ok:                   # re-decodes from the last keyframe (10x slower)
            break
        t_cur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_cur + 1e-3 >= next_t:   # first frame at/after each sample time
            times.append(round(t_cur, 2))
            dets.append(_detect(net, cv2, grids, strides, fr))
            next_t = times[-1] + step
        if time.time() - said > 30:
            said = time.time()
            print(f"   context: scanning... {len(times)} frames so far")
    cap.release()
    if len(times) < 4:
        return None
    d = {"meta": meta, "times": times, "dets": dets}
    cache.write_text(json.dumps(d), encoding="utf-8")
    print(f"   context: scanned {len(times)} frames in {time.time() - t0:.0f}s (cached)")
    return d


def _detect(net, cv2, grids, strides, frame):
    """One frame → kept detections, positions as shares of the frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ratio = min(640 / rgb.shape[0], 640 / rgb.shape[1])
    rh, rw = int(rgb.shape[0] * ratio), int(rgb.shape[1] * ratio)
    padded = np.full((640, 640, 3), 114.0, np.float32)
    padded[:rh, :rw] = cv2.resize(rgb, (rw, rh)).astype(np.float32)
    net.setInput(padded.transpose(2, 0, 1)[None])
    d = np.asarray(net.forward(net.getUnconnectedOutLayersNames())[0], np.float32)
    if d.ndim == 3:
        d = d[0]
    d[:, :2] = (d[:, :2] + grids) * strides
    d[:, 2:4] = np.exp(d[:, 2:4]) * strides
    scores = d[:, 4:5] * d[:, 5:]
    keep = np.amax(scores, axis=1) >= CONF
    if not np.any(keep):
        return []
    boxes = (d[keep, :4] / ratio).tolist()
    scores = scores[keep]
    cls = np.argmax(scores, axis=1)
    conf = np.amax(scores, axis=1)
    nms = (cv2.dnn.NMSBoxesBatched if hasattr(cv2.dnn, "NMSBoxesBatched")
           else cv2.dnn.NMSBoxes)(boxes, conf.tolist(), cls.tolist(), CONF, 0.5)
    h, w = frame.shape[:2]
    out = []
    for i in np.array(nms).flatten():
        x, y, bw, bh = boxes[i]
        out.append([CLASSES[cls[i]], round(float(conf[i]), 2),
                    round((x + bw / 2) / w, 3), round((y + bh / 2) / h, 3),
                    round(bw / w, 3), round(bh / h, 3)])
    return out


# ---------------- what was said ----------------

def mentions(words, s, e):
    """(t_start, t_end, word, classes) for every mapped keyword spoken in [s, e]."""
    toks = [(w["w"].lower().strip("\"'.,!?…"), w["s"], w["e"])
            for w in words if s <= w["s"] < e]
    out = []
    for i, (tok, ts, te) in enumerate(toks):
        hit = _WORDS.get(tok)
        if hit is None and i + 1 < len(toks):
            hit = _PHRASES.get(f"{tok} {toks[i + 1][0]}")
            if hit is not None:
                te = toks[i + 1][2]
        if hit is not None:
            out.append((ts, te, tok, hit))
    return out


def _find(index, classes, t0, t1):
    """Detection samples of any of these classes inside [t0, t1]: (t, det)."""
    want = set(classes)
    return [(t, x) for t, d in zip(index["times"], index["dets"])
            if t0 <= t <= t1 for x in d if x[0] in want]


# ---------------- emphasis moves ----------------

def events(video, words, clip, idx, shots):
    """Emphasis moves for one clip: verified speech↔picture matches only.

    Every move is clipped inside one camera shot, starts at least a second
    into the clip, is separated from the last one, and the same thing is
    never emphasized twice in a clip.
    """
    s0, s1 = clip["start"], clip["end"]
    dur = s1 - s0
    w, h, cw, ch, y = C.crop_geometry(video)
    max_events = min(MAX_EVENTS, C.CTX_MAX_EVENTS)
    used, out = set(), []
    for ts, te, word, classes in mentions(words, s0, s1):
        if len(out) >= max_events:
            break
        key = "+".join(sorted(classes))
        if key in used:
            continue
        rel = ts - s0
        shot = next((x for x in shots if x[0] <= rel < x[1]), None)
        if shot is None or shot[2] != "crop" or not shot[3]:
            continue                                    # pad/split: play normally
        seen = _find(idx, classes, ts - LOOK_BACK, te + LOOK_AHEAD)
        seen = [x for x in seen if x[1][4] * x[1][5] >= MIN_BOX]
        if len(seen) < 2 and not (seen and seen[0][1][1] >= 0.6):
            continue                                    # not convincingly visible
        # event window, kept inside this shot and the clip's quiet edges
        t0 = max(rel - 0.35, shot[0] + 0.02, 1.0)
        t1 = min(seen[-1][0] - s0 + 0.45, rel + MAX_HOLD, shot[1] - 0.02, dur - 0.8)
        if t1 - t0 < 1.0:
            continue
        # face point and the object box at the mention
        fx, fy = min(shot[3], key=lambda p: abs(p[0] - t0))[1:]
        near = [x for x in seen if abs(x[0] - ts) < 0.9] or seen
        med = lambda k: sorted(x[1][k] for x in near)[len(near) // 2]
        mcx, mcy, mbw, mbh = med(2), med(3), med(4), med(5)
        # where the object sits relative to the tracked crop, over time
        def band_x(t):                                  # crop-window left edge
            px = _interp(shot[3], t)[0] if len(shot[3]) > 1 else shot[3][0][1]
            return max(0.0, min(px - cw / 2, w - cw))
        ax, ay = (fx - band_x(t0)) / cw, (fy - y) / ch
        kfs = []
        for kt, det in seen:
            kr = max(t0 + 0.45, min(kt - s0, t1 - 0.45))
            kx, ky = (det[2] * w - band_x(kr)) / cw, (det[3] * h - y) / ch
            kfs.append((kr, kx, ky))
        kfs = sorted(set(kfs))[:8] or [(t0 + 0.45, ax, ay)]
        # zoom toward it while the face stays in frame, else reframe instead
        sep_x = max(abs(k[1] - ax) for k in kfs)
        sep_y = max(abs(k[2] - ay) for k in kfs)
        z_geo = min(1 / (sep_x + 0.26), 1 / (sep_y + 0.26))
        area = mbw * mbh
        z_peak = 1.35 if area < 0.015 else (1.25 if area < 0.05 else 1.18)
        z = min(z_peak, z_geo)
        if z >= MIN_ZOOM:
            kind, extra = "zoom", {"z": round(z, 3), "face": (round(ax, 4), round(ay, 4)),
                                   "kfs": [(round(t, 3), round(x, 4), round(yy, 4))
                                           for t, x, yy in kfs]}
        else:
            kind, extra = "focus", {"focus_pts": [(round(fx), round(fy)),
                                                  (round(mcx * w), round(mcy * h))]}
        out.append(dict(t0=round(t0, 3), t1=round(t1, 3), kind=kind, cls=key,
                        conf=max(x[1][1] for x in seen), **extra))
        used.add(key)
    out.sort(key=lambda e: e["t0"])
    keep = []
    for e in out:                                       # one move at a time
        if keep and e["t0"] < keep[-1]["t1"] + C.CTX_GAP:
            continue
        keep.append(e)
    return keep


def _interp(pts, t):
    """Face position at t, linear between the shot's face samples."""
    pts = sorted(pts)
    if t <= pts[0][0]:
        return pts[0][1], pts[0][2]
    for (ta, xa, ya), (tb, xb, yb) in zip(pts, pts[1:]):
        if t <= tb:
            f = (t - ta) / max(tb - ta, 1e-6)
            return xa + (xb - xa) * f, ya + (yb - ya) * f
    return pts[-1][1], pts[-1][2]


def hook_match(idx, words, s, e):
    """Share of the hook's keywords that are verifiably on screen (for scoring)."""
    ms = mentions(words, s, e)[:4]
    if not ms:
        return 0.0
    hit = sum(1 for ts, te, _, classes in ms if _find(idx, classes, ts - LOOK_BACK, te + LOOK_AHEAD))
    return hit / len(ms)
