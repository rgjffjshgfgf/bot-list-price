"""The bot's own price detector and what it has learned so far.

Two kinds of knowledge, both learned from checked answers (Gemini's, or the
user's ✅):

* Price model – a logistic model over features of each number (its column
  header, its shape, its column's behaviour...). It starts from built-in prior
  knowledge and keeps learning, so it can handle list formats it never saw.
* Format memory – every list format seen (its words, its columns, which
  columns hold prices). A new file of a known format is handled exactly like
  last time.

Nothing is used on its own until it has proved itself: a format becomes
"trusted" after it matched the teacher's answer several times in a row, the
model after it got whole pages right many times in a row.
"""
from __future__ import annotations

import json
import logging
import math
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .layout import (CURRENCIES, PageDoc, _is_sequence, canon, digits_of, keyword_groups,
                     magnitude)

log = logging.getLogger(__name__)

# ================================================================ features ==

PRIOR: dict[str, float] = {
    "bias": -0.6,
    "kw:price": 3.0, "kw:code": -2.6, "kw:row": -3.0, "kw:qty": -2.6, "kw:date": -2.6, "kw:phone": -3.0,
    "grp:yes": 1.3, "grp:no": -1.3,
    "digits:1-3": -1.8, "digits:4-6": 0.3, "digits:7-9": 0.6, "digits:10+": -1.4,
    "tz:3+": 0.9, "tz:0": -0.5,
    "attached": -4.0, "yearlike": -1.2, "dec": -0.3,
    "col_code_like": -2.2, "col_seq": -4.0, "len_outlier": -0.6,
    "colgrp:hi": 0.8, "colgrp:lo": -0.4,
    "cur_adj": 1.5, "row_phone": -2.0,
}

_SEP_NAMES = {",": "comma", "٬": "comma", "،": "comma", "/": "slash", ".": "dot", "٫": "dot",
              "'": "quote", "’": "quote"}


def _bucket_digits(n: int) -> str:
    return "1-3" if n <= 3 else "4-6" if n <= 6 else "7-9" if n <= 9 else "10+"


def features(doc: PageDoc, i: int) -> list[str]:
    t = doc.nums[i]
    n = t.num
    c = doc.col_of[i]
    col = doc.columns[c]
    d = digits_of(t.text)
    f = ["bias", "digits:" + _bucket_digits(len(d))]
    grp = bool(n.fmt.group_sep)
    f.append("grp:yes" if grp else "grp:no")
    if grp:
        f.append("sep:" + _SEP_NAMES.get(n.fmt.group_sep, "space"))
    if n.fmt.decimals:
        f.append("dec")
    f.append("script:" + n.fmt.script)
    tz = len(d) - len(d.rstrip("0"))
    f.append("tz:" + ("0" if tz == 0 else "1-2" if tz < 3 else "3+"))
    f.append(f"mag:{min(magnitude(n.value), 12)}")
    if t.attached:
        f.append("attached")
    if not grp and len(d) == 4 and (1300 <= int(d) <= 1499 or 1990 <= int(d) <= 2100):
        f.append("yearlike")

    # the column this number sits in
    ncol = len(col)
    f.append("coln:" + ("1" if ncol == 1 else "2-3" if ncol <= 3 else "4-9" if ncol < 10 else "10+"))
    grp_frac = sum(1 for k in col if doc.nums[k].num.fmt.group_sep) / ncol
    f.append("colgrp:" + ("hi" if grp_frac >= 0.8 else "mid" if grp_frac >= 0.3 else "lo"))
    lens = [len(digits_of(doc.nums[k].text)) for k in col]
    if ncol >= 3 and grp_frac == 0 and len(set(lens)) == 1 and lens[0] >= 5:
        f.append("col_code_like")
    if _is_sequence(doc, c):
        f.append("col_seq")
    if ncol >= 3 and abs(len(d) - statistics.median(lens)) >= 2:
        f.append("len_outlier")
    f.append(f"rr:{min(doc.col_rank_right(c), 4)}")
    f.append(f"rl:{min(doc.col_rank_left(c), 4)}")
    if ncol >= 3 and sum(1 for k in col if digits_of(doc.nums[k].text).endswith("000")) >= 0.7 * ncol:
        f.append("colk:hi")

    # what the column / cell is called
    hdr = doc.header(c)
    ab = doc.above(i)
    groups: set[str] = set()
    for w in hdr + ab:
        groups |= keyword_groups(w.text)
    f += ["kw:" + g for g in sorted(groups)]
    if not hdr:
        f.append("nohdr")
    f += ["h:" + cw for cw in {canon(w.text) for w in hdr[:6]} if cw]
    f += ["a:" + cw for cw in {canon(w.text) for w in ab[:3]} if cw]

    # the row
    words, _ = doc.row(i)
    for w in words:
        if CURRENCIES.get(canon(w.text)) and min(abs(w.box[0] - t.box[2]), abs(t.box[0] - w.box[2])) <= 5 * doc.line_h:
            f.append("cur_adj")
            break
    if any("phone" in keyword_groups(w.text) for w in words):
        f.append("row_phone")
    return f


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


# =================================================================== model ==

class PriceModel:
    """Logistic regression on top of the prior weights (learned part starts at 0)."""

    def __init__(self, weights: dict | None = None, g2: dict | None = None):
        self.w: dict[str, float] = dict(weights or {})
        self.g2: dict[str, float] = dict(g2 or {})

    def logit(self, feats: list[str]) -> float:
        return sum(PRIOR.get(k, 0.0) + self.w.get(k, 0.0) for k in feats)

    def predict(self, doc: PageDoc) -> list[float]:
        z = [self.logit(features(doc, i)) for i in range(len(doc.nums))]
        out = list(z)
        for col in doc.columns:
            if len(col) >= 3:
                m = sum(z[i] for i in col) / len(col)
                for i in col:
                    out[i] = 0.7 * z[i] + 0.3 * m
        return [_sigmoid(v) for v in out]

    def train(self, samples: list[tuple[list[str], int]], epochs: int = 4, lr: float = 0.5,
              l2: float = 1e-4) -> None:
        """AdaGrad; the learned weights are pulled toward 0 = toward the priors."""
        data = list(samples)
        for _ in range(epochs):
            random.shuffle(data)
            for feats, y in data:
                p = _sigmoid(self.logit(feats))
                g = p - y
                for k in feats:
                    wk = self.w.get(k, 0.0)
                    grad = g + l2 * wk
                    acc = self.g2.get(k, 0.0) + grad * grad
                    self.g2[k] = acc
                    wk -= lr * grad / (math.sqrt(acc) + 1e-8)
                    if abs(wk) < 1e-6:
                        self.w.pop(k, None)
                    else:
                        self.w[k] = max(-12.0, min(12.0, wk))


# =============================================================== templates ==

def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _span_overlap(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def _col_record(doc: PageDoc, c: int, idx: list[int]) -> dict:
    x0 = min(doc.nums[i].box[0] for i in idx) / doc.width
    x1 = max(doc.nums[i].box[2] for i in idx) / doc.width
    lens = [len(digits_of(doc.nums[i].text)) for i in idx]
    return {"span": [round(x0, 4), round(x1, 4)],
            "hdr": sorted({canon(w.text) for w in doc.header(c)} - {""}),
            "grp": all(bool(doc.nums[i].num.fmt.group_sep) for i in idx),
            "len": [min(lens), max(lens)]}


def _header_words(doc: PageDoc) -> set[str]:
    out: set[str] = set()
    for c in range(len(doc.columns)):
        if len(doc.columns[c]) >= 2:
            out |= {w for w in (canon(t.text) for t in doc.header(c)) if len(w) >= 2 and not any(ch.isdigit() for ch in w)}
    return out


def template_predict(tpl: dict, doc: PageDoc) -> set[int]:
    sel: set[int] = set()
    cols = [(True, pc) for pc in tpl.get("cols", [])] + [(False, pc) for pc in tpl.get("neg", [])]
    if not cols:
        return sel
    for c in range(len(doc.columns)):
        hdr = {canon(w.text) for w in doc.header(c)} - {""}
        span = list(doc.norm_span(c))
        best, best_s = None, 0.0
        for is_price, pc in cols:
            s = _span_overlap(span, pc["span"])
            if hdr and pc["hdr"]:
                s += 1.5 * _jaccard(hdr, set(pc["hdr"]))
            if s > best_s:
                best, best_s = (is_price, pc), s
        if best is None or not best[0] or best_s < 0.6:
            continue
        pc = best[1]
        lo, hi = pc["len"]
        for i in doc.columns[c]:
            n = doc.nums[i]
            if n.attached or (pc["grp"] and not n.num.fmt.group_sep):
                continue
            if lo - 1 <= len(digits_of(n.text)) <= hi + 2:
                sel.add(i)
    return sel


# =================================================================== store ==

@dataclass
class Decision:
    selected: set[int]
    probs: list[float]
    how: str                        # "format" | "model"
    trusted: bool
    template: dict | None = None
    similarity: float = 0.0
    confident: bool = False
    model_selected: set[int] = field(default_factory=set)


class Brain:
    SAMPLE_CAP = 40000
    HISTORY = 60

    def __init__(self, directory: Path, trust_after: int = 2, general_after: int = 20):
        self.dir = directory
        self.trust_after = max(1, trust_after)
        self.general_after = max(3, general_after)
        self.lock = threading.RLock()
        self.model = PriceModel()
        self.templates: list[dict] = []
        self.stats: dict = {}
        self.samples: list[tuple[list[str], int]] = []
        self._load()

    # ---- persistence ------------------------------------------------------
    @property
    def _main(self) -> Path:
        return self.dir / "brain.json"

    @property
    def _samples_path(self) -> Path:
        return self.dir / "samples.jsonl"

    def _load(self) -> None:
        try:
            data = json.loads(self._main.read_text(encoding="utf-8"))
            self.model = PriceModel(data.get("weights"), data.get("g2"))
            self.templates = data.get("templates", [])
            self.stats = data.get("stats", {})
        except (OSError, ValueError):
            pass
        try:
            with self._samples_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        s = json.loads(line)
                        self.samples.append((s["f"], int(s["y"])))
                    except (ValueError, KeyError, TypeError):
                        continue
            self.samples = self.samples[-self.SAMPLE_CAP:]
        except OSError:
            pass

    def merge_seed(self, seed_dir: Path) -> int:
        """Add lessons shipped with the code (taught and checked by hand) that this
        brain has not got yet. Returns the number of new formats."""
        try:
            data = json.loads((seed_dir / "brain.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if seed_dir.resolve() == self.dir.resolve():
            return 0
        with self.lock:
            done = set(self.stats.get("seed_formats", []))
            have = {t["id"] for t in self.templates}
            new = [t for t in data.get("templates", []) if t["id"] not in have and t["id"] not in done]
            if not new and done:
                return 0
            self.templates.extend(new)
            self.stats["seed_formats"] = sorted(done | {t["id"] for t in data.get("templates", [])})
            if not self.model.w:
                self.model = PriceModel(data.get("weights"), data.get("g2"))
            try:
                seed_samples = []
                with (seed_dir / "samples.jsonl").open(encoding="utf-8") as fh:
                    for line in fh:
                        s = json.loads(line)
                        seed_samples.append((s["f"], int(s["y"])))
            except (OSError, ValueError, KeyError):
                seed_samples = []
            if new and seed_samples and self.samples:
                self._append_samples(seed_samples)
                self.model.train(seed_samples + random.sample(self.samples, min(len(self.samples), 3000)), epochs=3)
            elif seed_samples and not self.samples:
                self._append_samples(seed_samples)
            self.save()
            return len(new)

    def save(self) -> None:
        with self.lock:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                data = {"version": 1, "weights": self.model.w, "g2": self.model.g2,
                        "templates": self.templates, "stats": self.stats}
                tmp = self._main.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self._main)
            except OSError as exc:
                log.warning("could not save brain: %s", exc)

    def _append_samples(self, new: list[tuple[list[str], int]]) -> None:
        self.samples.extend(new)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            if len(self.samples) > self.SAMPLE_CAP * 1.2:
                self.samples = self.samples[-self.SAMPLE_CAP:]
                tmp = self._samples_path.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps({"f": f, "y": y}, ensure_ascii=False) + "\n"
                                       for f, y in self.samples), encoding="utf-8")
                tmp.replace(self._samples_path)
            else:
                with self._samples_path.open("a", encoding="utf-8") as fh:
                    for f, y in new:
                        fh.write(json.dumps({"f": f, "y": y}, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("could not save samples: %s", exc)

    def reset(self) -> None:
        with self.lock:
            self.model = PriceModel()
            self.templates, self.stats, self.samples = [], {}, []
            for p in (self._main, self._samples_path):
                try:
                    p.unlink()
                except OSError:
                    pass

    # ---- matching ---------------------------------------------------------
    def match(self, doc: PageDoc) -> tuple[dict | None, float]:
        """The known format this page belongs to: same words overall, or the same
        column headers (other pages of a long list carry other products)."""
        sig = doc.signature()
        hdr = _header_words(doc)
        need = 0.55 if doc.source == "pdf" else 0.35
        best, best_s = None, 0.0
        with self.lock:
            for tpl in self.templates:
                if tpl.get("source") != doc.source:
                    continue
                s = _jaccard(sig, set(tpl.get("sig", [])))
                th = set(tpl.get("hdr", []))
                common = len(hdr & th)
                if common >= 2:
                    s = max(s, need + 0.4 * (common / min(len(hdr), len(th)) - 0.75))
                if s > best_s:
                    best, best_s = tpl, s
        return (best, best_s) if best is not None and best_s >= need else (None, best_s)

    def _hist(self, source: str) -> list[int]:
        return self.stats.setdefault("hist", {}).setdefault(source, [])

    def general_ready(self, source: str) -> bool:
        """The model got enough whole pages right in a row to handle unseen formats."""
        h = self._hist(source)[-self.general_after:]
        return len(h) >= self.general_after and sum(h) >= 0.95 * len(h) and all(h[-5:])

    def reads_ready(self, tpl: dict | None) -> bool:
        """Local reading of pixels is proven (photos/scans only)."""
        r = (tpl or {}).get("reads") or self.stats.get("reads", {})
        n, ok = r.get("n", 0), r.get("ok", 0)
        return n >= 20 and ok >= 0.99 * n

    def decide(self, doc: PageDoc, mode: str = "auto") -> Decision:
        probs = self.model.predict(doc) if doc.nums else []
        model_sel = {i for i, p in enumerate(probs) if p >= 0.5}
        confident = bool(probs) and all(p <= 0.15 or p >= 0.85 for p in probs)
        tpl, sim = self.match(doc)
        if tpl is not None:
            sel = template_predict(tpl, doc)
            trusted = tpl.get("streak", 0) >= self.trust_after
            if doc.source == "ocr":
                trusted = trusted and self.reads_ready(tpl)
            # the format memory and the model must not clearly disagree (layout changed?)
            strong = sum(1 for i, p in enumerate(probs) if (i in sel) != (p >= 0.5) and (p < 0.1 or p > 0.9))
            if strong > max(1, 0.05 * len(doc.nums)) or (not sel and model_sel):
                trusted = False
            dec = Decision(sel, probs, "format", trusted, tpl, sim, confident, model_sel)
        else:
            trusted = confident and bool(model_sel) and self.general_ready(doc.source)
            if doc.source == "ocr":
                trusted = trusted and self.reads_ready(None)
            dec = Decision(model_sel, probs, "model", trusted, None, sim, confident, model_sel)
        if mode == "teacher":
            dec.trusted = False
        elif mode == "local" and (dec.selected or not doc.nums):
            dec.trusted = True
        return dec

    # ---- learning ---------------------------------------------------------
    def learn(self, doc: PageDoc, truth: set[int], decision: Decision | None, teacher: str,
              extra: dict | None = None) -> dict:
        """Learn one checked page. `truth` = indices of doc.nums that are prices.
        Returns what happened (for the user-facing report)."""
        if not doc.nums:
            return {}
        with self.lock:
            samples = [(features(doc, i), 1 if i in truth else 0) for i in range(len(doc.nums))]
            replay = random.sample(self.samples, min(len(self.samples), 3000))
            self._append_samples(samples)
            self.model.train(samples + replay, epochs=3)
            self.model.train(samples, epochs=2)

            model_ok = decision is not None and decision.model_selected == truth
            hist = self._hist(doc.source)
            if decision is not None:
                hist.append(1 if model_ok else 0)
                del hist[:-self.HISTORY]
            tot = self.stats.setdefault("checked", {"pages": 0, "tokens": 0, "tokens_ok": 0})
            tot["pages"] += 1
            tot["tokens"] += len(doc.nums)
            if decision is not None:
                tot["tokens_ok"] += sum(1 for i in range(len(doc.nums)) if (i in decision.model_selected) == (i in truth))

            tpl, _ = self.match(doc)
            outcome = {"teacher": teacher, "model_ok": model_ok}
            if tpl is None:
                if truth:
                    tpl = {"id": f"f{int(time.time() * 1000) % 10**10}", "source": doc.source,
                           "name": doc.title() or "بدون عنوان", "created": time.time(), "streak": 0,
                           "checks": 0, "agree": 0, "seen": 0}
                    self.templates.append(tpl)
                    outcome["new_format"] = True
            else:
                before = template_predict(tpl, doc)
                agree = before == truth
                tpl["checks"] = tpl.get("checks", 0) + 1
                if agree:
                    tpl["agree"] = tpl.get("agree", 0) + 1
                    tpl["streak"] = tpl.get("streak", 0) + 1
                else:
                    tpl["streak"] = 0
                outcome["format_ok"] = agree
            if tpl is not None:
                self._fill_template(tpl, doc, truth)
                if extra and "img" in extra:
                    tpl["img"] = bool(extra["img"])
                tpl["seen"] = tpl.get("seen", 0) + 1
                tpl["updated"] = time.time()
                if extra and "reads" in extra:
                    r = tpl.setdefault("reads", {"n": 0, "ok": 0})
                    r["n"] += extra["reads"][0]
                    r["ok"] += extra["reads"][1]
                outcome["format"] = tpl
            if extra and "reads" in extra:
                r = self.stats.setdefault("reads", {"n": 0, "ok": 0})
                r["n"] += extra["reads"][0]
                r["ok"] += extra["reads"][1]
            self.templates = self.templates[-300:]
            self.save()
            return outcome

    def _fill_template(self, tpl: dict, doc: PageDoc, truth: set[int]) -> None:
        cols, neg = [], []
        for c, idx in enumerate(doc.columns):
            prices = [i for i in idx if i in truth]
            if prices:
                cols.append(_col_record(doc, c, prices))
            elif len(idx) >= 2:
                neg.append(_col_record(doc, c, idx))
        tpl["cols"], tpl["neg"] = cols, neg
        tpl["sig"] = sorted(doc.signature())[:400]
        tpl["hdr"] = sorted(_header_words(doc))[:60]
        if doc.title():
            tpl["name"] = doc.title()

    def feedback(self, template_ids: set[str], good: bool, wrong_models: list[str] = ()) -> None:
        with self.lock:
            for source in wrong_models:
                h = self._hist(source)
                h.append(0)
                del h[:-self.HISTORY]
            for tpl in self.templates:
                if tpl["id"] in template_ids:
                    if good:
                        tpl["streak"] = tpl.get("streak", 0) + 1
                        tpl["agree"] = tpl.get("agree", 0) + 1
                        tpl["checks"] = tpl.get("checks", 0) + 1
                    else:
                        tpl["streak"] = 0
                        tpl["checks"] = tpl.get("checks", 0) + 1
            fb = self.stats.setdefault("feedback", {"good": 0, "bad": 0})
            fb["good" if good else "bad"] += 1
            self.save()

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            day = time.strftime("%Y-%m-%d")
            c = self.stats.setdefault("counts", {})
            if c.get("day") != day:
                c.clear()
                c["day"] = day
            c[key] = c.get(key, 0) + n
            tot = self.stats.setdefault("totals", {})
            tot[key] = tot.get(key, 0) + n
            self.save()
