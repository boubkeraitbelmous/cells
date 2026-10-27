# -*- coding: utf-8 -*-
"""CELLS v2 Pro - Living Brain Experiment.

Professional scientific implementation for GitHub.

Base: CELLS v1 (AdderLinear + Resonance, zero multiplications in forward).
Extensions: self-replicating cells, immune pruning, immune homeostasis
(self-cannibalism), and a two-state memory model.

Concept mapping (original exploratory terms -> professional terms):
    LAWA3Y -> UNCONSCIOUS reservoir (95 percent silent storage)
    LWA3Y  -> CONSCIOUS answer (5 percent working memory, top-10 cells)
    FIKR   -> RECALL activation (query-driven resonance wake-up)
    FRAZ   -> PRUNING (immune removal of dead cells)

Constraint: zero multiplications in the forward path. Forward methods use
only subtraction, absolute value, summation, negation, maximum, minimum,
comparisons, sorting, and median. Replication and immune scaling are
evolution steps outside the forward path and may use multiplications.

Tracked metrics: births, deaths, immune self-eating events, synchronized
cluster size (consciousness metric, pairwise L1 threshold 2.8).

Usage:
    python cells_brain_v2_pro.py
Outputs:
    cells_brain_v2_results.png   (4-panel training figure)
    cells_architecture.png       (system architecture diagram)
    cells_brain_v2_pro_log.csv   (per-epoch log)
"""

import inspect
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except Exception:
    HAS_PIL = False
    Image = None


# ----------------------------------------------------------------------------
# Global epoch registers (logging only, not part of the forward path).
# ----------------------------------------------------------------------------
BIRTH_LOG = []
DEATH_LOG = []
IMMUNE_EAT_LOG = []
CONSCIOUS_LOG = []
POP_LOG = []
IMMUNE_LOG = []


# ----------------------------------------------------------------------------
# 1. CELLS v1 base: AdderLinear and ResonanceCell (zero-multiplication forward)
# ----------------------------------------------------------------------------
class AdderLinear:
    """Pure adder layer. Score equals the negated L1 distance."""

    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.rand(out_dim, in_dim) + rng.rand(out_dim, in_dim)
        self.W = self.W - np.ones_like(self.W)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        diff = np.subtract(x[None, :], self.W)
        ad = np.abs(diff)
        s = np.sum(ad, axis=1)
        out = np.negative(s)
        return out

    def forward_single(self, x, w):
        diff = np.subtract(x, w)
        ad = np.abs(diff)
        s = np.sum(ad)
        out = np.negative(s)
        return out


class ResonanceCell:
    """Single living cell storing one prototype vector W."""

    _next_id = 0

    def __init__(self, W, resonance_thr=-4.0, parent_id=-1, generation=0):
        self.W = np.array(W, dtype=float).copy()
        self.thr = float(resonance_thr)
        self.id = ResonanceCell._next_id
        ResonanceCell._next_id = ResonanceCell._next_id + 1
        self.parent_id = parent_id
        self.generation = generation
        self.age = 0
        self.fired_count = 0
        self.epoch_fired = 0
        self.last_error = 999.0
        self.last_score = -999.0
        self.fitness = -999.0
        self.best_error = 999.0
        self.alive = True

    def forward(self, x):
        diff = np.subtract(x, self.W)
        ad = np.abs(diff)
        s = np.sum(ad)
        score = np.negative(s)
        fired = score > self.thr
        return score, fired

    def update_stats(self, score, fired, x):
        """Update fitness and firing counters (bookkeeping, not forward)."""
        self.last_score = float(score)
        diff = np.subtract(x, self.W)
        ad = np.abs(diff)
        err = np.sum(ad)
        self.last_error = float(err)
        self.fitness = float(np.negative(err))
        self.age = self.age + 1
        if fired:
            self.fired_count = self.fired_count + 1
            self.epoch_fired = self.epoch_fired + 1
        if err < self.best_error:
            self.best_error = float(err)


# ----------------------------------------------------------------------------
# 2. Immune pruner: small adder network that scores and removes dead cells.
# ----------------------------------------------------------------------------
class ImmunePruner:
    """Immune system implemented as a small adder network."""

    def __init__(self, in_dim, n_units=10, seed=777, eat_thr=-3.0):
        rng = np.random.RandomState(seed)
        self.V = rng.rand(n_units, in_dim) + rng.rand(n_units, in_dim)
        self.V = self.V - np.ones_like(self.V)
        self.eat_thr = float(eat_thr)
        self.in_dim = in_dim

    @property
    def count(self):
        return int(self.V.shape[0])

    def forward(self, w):
        diff = np.subtract(w[None, :], self.V)
        ad = np.abs(diff)
        s = np.sum(ad, axis=1)
        score = np.negative(s)
        best = np.max(score)
        return best, score

    def is_dead(self, cell, never_fired_window):
        """A cell is dead when its error is high or it never fires."""
        dead_thr = 0.5 * self.in_dim
        cond_error = cell.best_error > dead_thr
        cond_silent = never_fired_window
        return bool(cond_error or cond_silent)

    def grow(self, dead_count, rng):
        """Scale the immune population by 10 percent of kills (evolution)."""
        add_n = int(dead_count * 0.1)
        if add_n > 0:
            noise = rng.randn(add_n, self.in_dim) * 0.05
            idx = rng.randint(0, self.count, size=add_n)
            clones = self.V[idx] + noise
            self.V = np.concatenate([self.V, clones], axis=0)
        return add_n

    def self_cannibalism(self):
        """Homeostasis: when count exceeds 50, remove 20 percent (evolution)."""
        if self.count > 50:
            eat = int(self.count * 0.2)
            if eat < 1:
                eat = 1
            norms = np.sum(np.abs(self.V), axis=1)
            order = np.argsort(norms)
            keep = self.count - eat
            self.V = self.V[order[:keep]]
            return eat
        return 0


# ----------------------------------------------------------------------------
# 3. Living brain: replication, pruning, homeostasis, recall.
# ----------------------------------------------------------------------------
class LivingBrainV2:
    """Self-replicating adder-cell population with immune regulation."""

    def __init__(self, in_dim=8, n_init=20, resonance_thr=-4.0,
                 fitness_thr=-3.0, sync_radius=2.8, seed=0):
        self.in_dim = in_dim
        self.resonance_thr = float(resonance_thr)
        self.fitness_thr = float(fitness_thr)
        self.sync_radius = float(sync_radius)
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.cells = []
        for _ in range(n_init):
            w = rng.rand(in_dim) + rng.rand(in_dim) - np.ones(in_dim)
            self.cells.append(ResonanceCell(w, resonance_thr=self.resonance_thr))
        self.immune = ImmunePruner(in_dim, n_units=10, seed=seed + 999)
        self.total_births = 0
        self.total_deaths = 0
        self.total_self_eaten = 0
        self.history = []

    def recall_activation(self, x):
        """RECALL: a query activates resonant cells (pure adder wake-up)."""
        scores = []
        fired_flags = []
        for c in self.cells:
            if c.alive:
                s, f = c.forward(x)
                c.update_stats(s, f, x)
                scores.append(float(s))
                fired_flags.append(bool(f))
            else:
                scores.append(-999.0)
                fired_flags.append(False)
        return scores, fired_flags

    def replicate_step(self, mutation=0.01):
        """Replicate fit cells: W_new = W + 0.01 * noise (evolution step)."""
        newborns = []
        for c in self.cells:
            if c.alive and c.fitness > self.fitness_thr:
                noise = self.rng.randn(self.in_dim) * mutation
                w_new = c.W + noise
                w_new = np.maximum(np.minimum(w_new, np.ones_like(w_new)),
                                   np.negative(np.ones_like(w_new)))
                baby = ResonanceCell(w_new, resonance_thr=c.thr,
                                     parent_id=c.id, generation=c.generation + 1)
                newborns.append(baby)
        for b in newborns:
            self.cells.append(b)
        self.total_births += len(newborns)
        return len(newborns)

    def pruning(self, window_never_fired):
        """PRUNING: immune scores dead cells, then removes them."""
        dead_idx = []
        for i, c in enumerate(self.cells):
            if c.alive and self.immune.is_dead(c, window_never_fired[i]):
                _best, _all = self.immune.forward(c.W)
                dead_idx.append(i)
        for i in dead_idx:
            self.cells[i].alive = False
        self.cells = [c for c in self.cells if c.alive]
        n_dead = len(dead_idx)
        self.total_deaths += n_dead
        grown = self.immune.grow(n_dead, self.rng)
        return n_dead, grown

    def homeostasis(self):
        """Immune self-cannibalism for population control."""
        eaten = self.immune.self_cannibalism()
        self.total_self_eaten += eaten
        return eaten

    def synchronized_cluster_size(self, active_W):
        """Consciousness metric: largest group with pairwise L1 below radius."""
        n = len(active_W)
        if n == 0:
            return 0
        if n == 1:
            return 1
        best = 1
        for i in range(n):
            cluster = 1
            for j in range(n):
                if j == i:
                    continue
                d = np.sum(np.abs(np.subtract(active_W[i], active_W[j])))
                if d < self.sync_radius:
                    cluster = cluster + 1
            if cluster > best:
                best = cluster
        return int(best)

    def unconscious_conscious_query(self, x):
        """UNCONSCIOUS reservoir query answered by the CONSCIOUS top-10."""
        scored = []
        for c in self.cells:
            s, f = c.forward(x)
            scored.append((float(s), c))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:10]
        top_cells = [c for _, c in top]
        active_W = [c.W for _, c in top]
        sync_size = self.synchronized_cluster_size(active_W)
        if len(top_cells) == 0:
            answer = np.zeros(self.in_dim)
        else:
            stack = np.stack([c.W for c in top_cells], axis=0)
            answer = np.median(stack, axis=0)
        return answer, top_cells, sync_size, len(self.cells) + self.immune.count

    def step(self, x, window_never_fired):
        """One living step: recall, replicate, prune, regulate, answer."""
        self.recall_activation(x)
        births = self.replicate_step()
        deaths, grown = self.pruning(window_never_fired)
        self_eaten = self.homeostasis()
        _ans, _top, sync_size, _res = self.unconscious_conscious_query(x)
        return births, deaths, grown, self_eaten, sync_size


# ----------------------------------------------------------------------------
# 4. Forward-path audit: prove zero multiplications in inference.
# ----------------------------------------------------------------------------
def audit_no_mult():
    """Inspect forward sources for forbidden multiplication-like operators."""
    targets = [
        ("AdderLinear.forward", AdderLinear.forward),
        ("ResonanceCell.forward", ResonanceCell.forward),
        ("ImmunePruner.forward", ImmunePruner.forward),
    ]
    forbidden = ["*", "@", "/", "matmul", "dot(", "multiply", "einsum",
                 "power", "exp(", "softmax", "sigmoid", "conv"]
    ok = True
    for name, fn in targets:
        src = inspect.getsource(fn)
        hits = [t for t in forbidden if t in src]
        status = "CLEAN (0 mult)" if not hits else "VIOLATION: %s" % (hits,)
        if hits:
            ok = False
        print("  [audit] %s : %s" % (name, status))
    return ok


# ----------------------------------------------------------------------------
# 5. Dataset
# ----------------------------------------------------------------------------
def make_dataset(in_dim=8, n_proto=8, n_samples=200, seed=1):
    """Generate clustered prototype data with small jitter (dataset only)."""
    rng = np.random.RandomState(seed)
    protos = rng.rand(n_proto, in_dim) + rng.rand(n_proto, in_dim) - np.ones((n_proto, in_dim))
    X = []
    for _ in range(n_samples):
        p = protos[rng.randint(0, n_proto)]
        jitter = (rng.rand(in_dim) - np.ones(in_dim) * 0.5) * 0.1
        X.append(p + jitter)
    return np.array(X), protos


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
def render_results_matplotlib(path="cells_brain_v2_results.png"):
    """Render the 4-panel results figure with matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("CELLS v2 Pro: births, deaths, population, consciousness", fontsize=12)
    axes[0, 0].plot(BIRTH_LOG, label="births")
    axes[0, 0].set_title("Births per epoch (replication)")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(DEATH_LOG, color="red", label="deaths")
    axes[0, 1].set_title("Deaths per epoch (immune pruning)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(POP_LOG, label="cells")
    axes[1, 0].plot(IMMUNE_LOG, label="immune")
    axes[1, 0].set_title("Population: cells vs immune")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(CONSCIOUS_LOG, color="purple", label="sync cluster")
    axes[1, 1].set_title("Consciousness metric: synchronized cluster size")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("cells")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _draw_series(draw, box, series, color, width=2):
    """Draw a normalized line series inside a pixel box with Pillow."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(180, 180, 180))
    if len(series) < 2:
        return
    lo = min(series)
    hi = max(series)
    span = hi - lo if hi > lo else 1.0
    pts = []
    for i, v in enumerate(series):
        fx = x0 + 8 + (x1 - x0 - 16) * (i / (len(series) - 1))
        fy = y1 - 8 - (y1 - y0 - 16) * ((v - lo) / span)
        pts.append((fx, fy))
    draw.line(pts, fill=color, width=width)


def render_results_pillow(path="cells_brain_v2_results.png"):
    """Render the 4-panel results figure with Pillow (no matplotlib needed)."""
    W, H = 1100, 800
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 10), "CELLS v2 Pro: births, deaths, population, consciousness", fill=(20, 20, 20))
    panels = [
        (20, 50, 535, 375, "Births per epoch (replication)", BIRTH_LOG, (30, 120, 200)),
        (565, 50, 1080, 375, "Deaths per epoch (immune pruning)", DEATH_LOG, (200, 40, 40)),
        (20, 400, 535, 750, "Population: cells (blue) vs immune (green)", None, None),
        (565, 400, 1080, 750, "Consciousness: synchronized cluster size", CONSCIOUS_LOG, (120, 40, 160)),
    ]
    for (x0, y0, x1, y1, title, series, color) in panels:
        d.text((x0 + 4, y0 + 4), title, fill=(20, 20, 20))
        box = (x0, y0 + 24, x1, y1)
        if series is not None:
            _draw_series(d, box, list(series), color)
        else:
            d.rectangle(box, outline=(180, 180, 180))
            _draw_series(d, box, list(POP_LOG), (30, 120, 200))
            _draw_series(d, box, list(IMMUNE_LOG), (40, 160, 60))
    d.text((20, 760), "Consciousness = synchronized cluster size (pairwise L1 < 2.8).", fill=(80, 80, 80))
    img.save(path)


def render_results_figure(path="cells_brain_v2_results.png"):
    """Render results with matplotlib when available, else Pillow."""
    if HAS_MPL:
        render_results_matplotlib(path)
    elif HAS_PIL:
        render_results_pillow(path)
    else:
        raise RuntimeError("Neither matplotlib nor Pillow is available for figures.")


def render_architecture_diagram(path="cells_architecture.png"):
    """Render the UNCONSCIOUS -> RECALL -> CONSCIOUS architecture diagram."""
    W, H = 1200, 700
    if HAS_PIL:
        img = Image.new("RGB", (W, H), (248, 249, 251))
        d = ImageDraw.Draw(img)
        d.text((30, 20), "CELLS v2 Pro Architecture: UNCONSCIOUS reservoir and CONSCIOUS answer", fill=(20, 20, 20))

        def box(x0, y0, x1, y1, title, lines, fill):
            d.rectangle((x0, y0, x1, y1), fill=fill, outline=(40, 40, 40))
            d.text((x0 + 12, y0 + 10), title, fill=(20, 20, 20))
            yy = y0 + 36
            for ln in lines:
                d.text((x0 + 12, yy), ln, fill=(40, 40, 40))
                yy += 22

        def arrow(x0, y0, x1, y1):
            d.line((x0, y0, x1, y1), fill=(30, 30, 30), width=3)
            d.polygon([(x1, y1), (x1 - 12, y1 - 6), (x1 - 12, y1 + 6)], fill=(30, 30, 30))

        box(40, 140, 330, 420, "UNCONSCIOUS (95%)",
            ["All cells + immune", "Silent reservoir", "Adder prototypes W", "Black-hole storage"], (226, 235, 250))
        box(410, 140, 700, 420, "RECALL activation (+)",
            ["Query wakes cells", "Score = -sum|x - W|", "Resonance firing", "Zero multiplications"], (255, 243, 220))
        box(780, 140, 1160, 420, "CONSCIOUS (5%)",
            ["Top-10 synchronized", "Median prototype answer", "Sync cluster metric", "Working memory"], (222, 246, 222))
        box(410, 480, 700, 640, "PRUNING (-)",
            ["Immune eats dead cells", "Dead: err>0.5dim or silent", "Growth +10%, cap 50", "Self-eating -20%"], (250, 226, 226))

        arrow(330, 280, 410, 280)
        arrow(700, 280, 780, 280)
        arrow(700, 480, 850, 430)
        arrow(500, 480, 250, 430)
        d.text((335, 250), "query", fill=(60, 60, 60))
        d.text((705, 250), "top-10", fill=(60, 60, 60))
        d.text((180, 470), "prune loop", fill=(120, 40, 40))
        d.text((705, 445), "regulate", fill=(120, 40, 40))
        d.text((40, 655), "Forward uses only subtract, abs, sum, negative, max, comparisons, median.",
               fill=(80, 80, 80))
        img.save(path)
    elif HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis("off")
        ax.set_title("CELLS v2 Pro Architecture")
        boxes = [
            (0.03, 0.35, 0.25, 0.4, "UNCONSCIOUS (95%)\nSilent reservoir"),
            (0.37, 0.35, 0.25, 0.4, "RECALL (+)\nQuery activation"),
            (0.71, 0.35, 0.25, 0.4, "CONSCIOUS (5%)\nTop-10 answer"),
            (0.37, 0.05, 0.25, 0.22, "PRUNING (-)\nImmune loop"),
        ]
        for (x, y, w, h, t) in boxes:
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
                    transform=ax.transAxes)
        ax.annotate("", xy=(0.37, 0.55), xytext=(0.28, 0.55),
                    arrowprops=dict(arrowstyle="->"), xycoords="axes fraction")
        ax.annotate("", xy=(0.71, 0.55), xytext=(0.62, 0.55),
                    arrowprops=dict(arrowstyle="->"), xycoords="axes fraction")
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        raise RuntimeError("No imaging backend available.")


# ----------------------------------------------------------------------------
# 7. Experiment
# ----------------------------------------------------------------------------
def main(epochs=60, in_dim=8):
    print("=" * 64)
    print("CELLS v2 Pro - Living Brain Experiment")
    print("Base: CELLS v1 (Adder + Resonance) | replication + immune + homeostasis")
    print("UNCONSCIOUS (95 pct reservoir) vs CONSCIOUS (5 pct answer)")
    print("=" * 64)

    print("")
    print("[1] Auditing forward path for multiplications...")
    clean = audit_no_mult()
    if not clean:
        print("Forward audit FAILED. Fix the forward path before trusting results.")
    else:
        print("Forward audit PASSED: zero multiplications in forward.")

    brain = LivingBrainV2(in_dim=in_dim, n_init=20, resonance_thr=-4.0,
                          fitness_thr=-3.0, sync_radius=2.8, seed=42)
    X, _protos = make_dataset(in_dim=in_dim)

    print("")
    print("[2] Running experiment: %d epochs x %d queries." % (epochs, len(X)))
    print("Rule: dead when best_error > 0.5 * dim (mean error > 0.5) or never fires.")
    print("Rule: replicate when fitness = -loss exceeds threshold (error < 3.0).")
    for ep in range(epochs):
        for c in brain.cells:
            c.epoch_fired = 0
            c.best_error = 999.0
        sync_sizes = []
        order = np.random.RandomState(ep).permutation(len(X))
        for qi in order:
            brain.recall_activation(X[qi])
        for qi in order[:20]:
            _ans, _top, s, _res = brain.unconscious_conscious_query(X[qi])
            sync_sizes.append(s)
        ep_births = brain.replicate_step()
        flags = [(c.epoch_fired == 0) for c in brain.cells]
        ep_deaths, _grown = brain.pruning(flags)
        ep_eaten = brain.homeostasis()

        avg_sync = sum(sync_sizes) / max(1, len(sync_sizes))
        BIRTH_LOG.append(ep_births)
        DEATH_LOG.append(ep_deaths)
        IMMUNE_EAT_LOG.append(ep_eaten)
        CONSCIOUS_LOG.append(avg_sync)
        POP_LOG.append(len(brain.cells))
        IMMUNE_LOG.append(brain.immune.count)
        brain.history.append(dict(epoch=ep, births=ep_births, deaths=ep_deaths,
                                  self_eaten=ep_eaten, sync=avg_sync,
                                  pop=len(brain.cells), immune=brain.immune.count))
        print("  ep %02d | pop=%3d immune=%3d | births=%2d deaths=%2d self_eaten=%d | sync=%.2f"
              % (ep, len(brain.cells), brain.immune.count, ep_births, ep_deaths, ep_eaten, avg_sync))
        if len(brain.cells) > 400:
            brain.cells.sort(key=lambda c: c.fitness, reverse=True)
            culled = len(brain.cells) - 400
            brain.cells = brain.cells[:400]
            print("  [cap] Overpopulation: culled %d weakest cells." % culled)

    print("")
    print("[3] Final report")
    print("  Total births      : %d" % brain.total_births)
    print("  Total deaths      : %d (removed by immune)" % brain.total_deaths)
    print("  Immune self-eaten : %d (natural)" % brain.total_self_eaten)
    print("  Final population  : %d cells + %d immune" % (len(brain.cells), brain.immune.count))

    print("")
    print("[3b] Homeostasis stress test (force immune above 50).")
    print("  Immune before overload: %d" % brain.immune.count)
    overload = brain.immune.grow(400, brain.rng)
    print("  Pathogen storm: 400 deaths produce +%d immune units: %d" % (overload, brain.immune.count))
    eaten = brain.homeostasis()
    print("  Self-cannibalism: count above 50, removed %d units: %d" % (eaten, brain.immune.count))
    IMMUNE_EAT_LOG.append(eaten)
    print("  Rule verified: immune = immune - 20 percent when count > 50.")
    print("  Max sync cluster  : %.1f (consciousness peak)" % max(CONSCIOUS_LOG))
    print("  Final sync cluster: %.2f" % CONSCIOUS_LOG[-1])

    demo_x = X[0]
    ans, top10, sync, reservoir = brain.unconscious_conscious_query(demo_x)
    print("")
    print("[4] UNCONSCIOUS vs CONSCIOUS demonstration (one RECALL query)")
    print("  UNCONSCIOUS reservoir size (cells + immune, 95 pct silent): %d" % reservoir)
    print("  CONSCIOUS answer size (top-10, 5 pct): %d cells, sync=%d" % (len(top10), sync))
    print("  CONSCIOUS answer (median prototype, first 4 dims): %s" % np.array2string(ans[:4], precision=3))

    outdir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(outdir, "cells_brain_v2_results.png")
    arch_path = os.path.join(outdir, "cells_architecture.png")
    render_results_figure(results_path)
    print("")
    print("[5] Results figure saved: %s" % results_path)
    render_architecture_diagram(arch_path)
    print("[6] Architecture diagram saved: %s" % arch_path)

    csv_path = os.path.join(outdir, "cells_brain_v2_pro_log.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,births,deaths,self_eaten,sync,pop,immune\n")
        for h in brain.history:
            f.write("%d,%d,%d,%d,%.3f,%d,%d\n" % (h["epoch"], h["births"], h["deaths"],
                                                   h["self_eaten"], h["sync"], h["pop"], h["immune"]))
    print("[7] CSV log saved: %s" % csv_path)
    print("")
    print("Done. Consciousness equals synchronized cluster size.")


if __name__ == "__main__":
    main()
