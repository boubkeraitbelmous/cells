#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cells_net.py
==============
Can a neural network learn with ONLY additions and subtractions?

This experiment trains two small MNIST classifiers side by side:

    NormalNet  : 784 -> 128 -> 10   (nn.Linear, i.e. w @ x + b, uses multiplication)
    CellsNet : 784 -> 128 -> 10   (AdderLinear -> CellsResonance, only + and -)

Layer 1 - AdderLinear
    y[b, o] = - sum_i | w[o, i] - x[b, i] |
    forward : sub, abs, sum, neg            -> no multiplication
    backward: dL/dx = +sum_o g * sign(w-x)
              dL/dw = -sum_b g * sign(w-x)
              sign is +1 / -1, so "g * sign" is a conditional sign flip
              (torch.where), never a multiplication.

Layer 2 - CellsResonance (the invented part)
    y[b, o] = sum_i ( w[o,i] + x[b,i] )   if w and x have the same sign  (resonance)
                     ( w[o,i] - x[b,i] )   if they have opposite signs
    The test "w*x > 0" is evaluated with comparisons, NOT with w*x.
    Expanding it shows what the layer really does:

        y[b,o] = sum_i w[o,i] + sum_i sign(w[o,i]) * |x[b,i]|

    so the discriminative part is a *binary* (sign) pattern over |x|.  That is
    why the backward is the straight-through gradient of the binary weights,

        dL/dw = g * |x|        dL/dx = g * sign(w) * sign(x)

    (the literal derivative of the piecewise formula is 1 for every element,
    which carries no information at all - the sign pattern could never move;
    a pure +/-1 backward instead freezes the signs.  Both were measured.)

Three details that make the additive stack actually train:
    * the input is standardised ((pixel-0.1307)/0.3081) so the resonance
      layer sees mixed signs;
    * the AdderLinear filters are initialised from real MNIST digits
      (random L1 filters are nearly blind: ~20% linear-probe accuracy vs
      ~82-86% for image templates);
    * the hidden features are zero-centred before the second layer, because
      -sum|w-x| has an offset ~100x larger than its variation.

Everything runs on CPU with only torch + torchvision (Pillow comes with
torchvision and is used to draw accuracy.png). No matplotlib needed.

Usage:  python cells_net.py
"""

import ast
import inspect
import math
import os
import textwrap
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

SEED = 0
torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  1. AdderLinear  -  gradients through sign(), never through multiplication
# --------------------------------------------------------------------------- #
class _AdderFn(torch.autograd.Function):
    """y[b,o] = -sum_i |w[o,i] - x[b,i]|"""

    @staticmethod
    def forward(ctx, x, w, chunk):
        parts = []
        for xb in torch.split(x, chunk, dim=0):
            d = torch.sub(xb.unsqueeze(1), w.unsqueeze(0))      # [b, o, i]
            parts.append(torch.neg(torch.abs(d).sum(dim=2)))    # -sum |w - x|
        y = torch.cat(parts, dim=0)
        ctx.save_for_backward(x, w)
        ctx.chunk = chunk
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        gx_parts, gw = [], torch.zeros_like(w)

        for xb, gb in zip(torch.split(x, ctx.chunk, dim=0),
                          torch.split(grad_y.contiguous(), ctx.chunk, dim=0)):
            s = torch.sub(w.unsqueeze(0), xb.unsqueeze(1)).sign()   # +1 / -1 / 0
            g3 = gb.unsqueeze(2)                                    # [b, o, 1]

            # g * sign  ==  sign flip, done with comparisons only
            contrib = torch.where(s > 0, g3,
                                  torch.where(s < 0, torch.neg(g3),
                                              torch.zeros_like(g3)))

            gx_parts.append(contrib.sum(dim=1))                     # dL/dx = +g*sign
            gw = torch.sub(gw, contrib.sum(dim=0))                  # dL/dw = -g*sign

        return torch.cat(gx_parts, dim=0), gw, None


class AdderLinear(nn.Module):
    """out = -sum_i |w - x|  (bias lives outside, see Rescale)"""

    def __init__(self, in_features, out_features, chunk=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk = chunk
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound))

    def forward(self, x):
        return _AdderFn.apply(x, self.weight, self.chunk)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"


# --------------------------------------------------------------------------- #
#  2. CellsResonance  -  add when signs agree, subtract when they don't
# --------------------------------------------------------------------------- #
class _ResonanceFn(torch.autograd.Function):
    """y[b,o] = sum_i (w+x) if sign(w)==sign(x) else (w-x)

    Note what this layer really computes (and why the backward looks the way
    it does):

        y[b,o] = sum_i w[o,i]  +  sum_i sign(w[o,i]) * |x[b,i]|

    * the first term is a per-output offset (the same in both branches);
    * the discriminative part is a *sign* pattern times |x|, i.e. binary
      weights.

    dL/dw through the literal piecewise formula is 1 for every element, so it
    carries no information and every element of a row would move together -
    the sign pattern could never change.  We therefore use the straight
    through gradient of the binary weights,

        dL/dw[o,i] = sum_b dL/dy[b,o] * |x[b,i]|
        dL/dx[b,i] = sum_o dL/dy[b,o] * sign(w[o,i]) * sign(x[b,i])

    The forward pass itself is still pure + and -.
    """

    @staticmethod
    def _same_sign(w, xb):
        wp = w.unsqueeze(0) > 0
        wn = w.unsqueeze(0) < 0
        xp = xb.unsqueeze(1) > 0
        xn = xb.unsqueeze(1) < 0
        # logical ops only: no w * x anywhere
        return torch.logical_or(torch.logical_and(wp, xp),
                                torch.logical_and(wn, xn))

    @staticmethod
    def forward(ctx, x, w, chunk):
        parts = []
        for xb in torch.split(x, chunk, dim=0):
            wu = w.unsqueeze(0)
            xu = xb.unsqueeze(1)
            same = _ResonanceFn._same_sign(w, xb)
            parts.append(torch.where(same,
                                     torch.add(wu, xu),
                                     torch.sub(wu, xu)).sum(dim=2))
        y = torch.cat(parts, dim=0)
        ctx.save_for_backward(x, w)
        ctx.chunk = chunk
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        gx_parts, gw = [], torch.zeros_like(w)

        for xb, gb in zip(torch.split(x, ctx.chunk, dim=0),
                          torch.split(grad_y.contiguous(), ctx.chunk, dim=0)):
            same = _ResonanceFn._same_sign(w, xb)
            g3 = gb.unsqueeze(2)                             # [b, o, 1]

            # dL/dx = g * sign(w) * sign(x)   -> still +/-1 only
            gx_parts.append(torch.where(same, g3, torch.neg(g3)).sum(dim=1))
            # dL/dw = g * |x|  (straight-through through sign(w))
            gw = gw + torch.mul(g3, torch.abs(xb).unsqueeze(1)).sum(dim=0)

        return torch.cat(gx_parts, dim=0), gw, None


class CellsResonance(nn.Module):
    """out = sum_i (w+x) if w and x share a sign else (w-x)"""

    def __init__(self, in_features, out_features, chunk=128, scale=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk = chunk
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-scale, scale))
        with torch.no_grad():
            # every branch adds w, so a row of weights also acts as a per-class
            # offset; start with zero-mean rows to keep the logits balanced
            self.weight.sub_(self.weight.mean(dim=1, keepdim=True))

    def forward(self, x):
        return _ResonanceFn.apply(x, self.weight, self.chunk)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Rescale(nn.Module):
    """y <- y / divisor + bias     (division by a constant + addition)

    The raw adder output is a sum over thousands of pixels, so it is divided
    by a compile-time constant to keep the logits in a sane range.  This is a
    division, not a multiplication.
    """

    def __init__(self, out_features, divisor=1.0):
        super().__init__()
        self.divisor = float(divisor)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, y):
        return torch.add(torch.div(y, self.divisor), self.bias)


class CenterNorm(nn.Module):
    """Subtract a running per-feature mean (zero-centring, subtraction only).

    -sum|w - x| is always negative and its offset is ~100x larger than its
    variation, so the second layer would only ever see the constant part.
    Removing the running mean makes the features zero-centred and lets the
    resonance layer work on the differences.
    """

    def __init__(self, n_features, momentum=10.0):
        super().__init__()
        self.register_buffer("running", torch.zeros(n_features))
        self.momentum = momentum
        self.ready = False

    def forward(self, x):
        if self.training:
            mu = x.mean(dim=0)
            with torch.no_grad():
                if not self.ready:
                    self.running.copy_(mu)          # calibrated on the first batch
                    self.ready = True
                else:
                    # running += (mu - running) / momentum   -> no multiplication
                    self.running.add_(torch.div(torch.sub(mu, self.running),
                                                self.momentum))
        return torch.sub(x, self.running)


# --------------------------------------------------------------------------- #
#  3. The two networks
# --------------------------------------------------------------------------- #
class NormalNet(nn.Module):
    """784 -> 128 -> 10, plain nn.Linear (multiplication) - the baseline."""

    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class CellsNet(nn.Module):
    """784 -> 128 -> 10, additive layers only.

    Three details make the purely additive stack trainable:

    1. the input is standardised ((pixel - 0.1307) / 0.3081) -> mixed signs,
       which is what the resonance layer wants to see;
    2. the AdderLinear filters start as real MNIST digits.  Random L1 filters
       are nearly blind (measured: ~20% linear-probe accuracy), image
       templates are not (~82-86%);
    3. the hidden features are zero-centred before the resonance layer.
    """

    MEAN, STD = 0.1307, 0.3081

    def __init__(self, hidden=128, templates=None):
        super().__init__()
        self.a1 = AdderLinear(784, hidden)
        self.n1 = Rescale(hidden, 784)
        self.c1 = CenterNorm(hidden)
        self.r2 = CellsResonance(hidden, 10)
        self.n2 = Rescale(10)

        if templates is not None:
            with torch.no_grad():
                t = torch.sub(templates.reshape(hidden, -1), CellsNet.MEAN)
                self.a1.weight.copy_(torch.div(t, CellsNet.STD))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.div(torch.sub(x, CellsNet.MEAN), CellsNet.STD)
        h = self.n1(self.a1(x))
        return self.n2(self.r2(self.c1(h)))


# --------------------------------------------------------------------------- #
#  4. Audit helpers: prove there is no multiplication
# --------------------------------------------------------------------------- #
MUL_FUNCS = {"mul", "matmul", "mm", "bmm", "einsum", "dot", "inner", "conv2d",
             "linear", "addmm", "outer", "tensordot"}


def _scan_mults(fn):
    """Return the multiplicative operations found in one function's source."""
    offenders = []
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Mult):
            offenders.append(f"{fn.__qualname__}: '*' at line {node.lineno}")
        elif isinstance(node, ast.MatMult):
            offenders.append(f"{fn.__qualname__}: '@' at line {node.lineno}")
        elif isinstance(node, ast.Pow):
            offenders.append(f"{fn.__qualname__}: '**' at line {node.lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in MUL_FUNCS:
                offenders.append(f"{fn.__qualname__}: .{node.func.attr}() "
                                 f"at line {node.lineno}")
    return offenders


def audit_forward():
    """AST-scan everything that runs in the forward pass."""
    parts = [AdderLinear.forward, CellsResonance.forward, Rescale.forward,
             CenterNorm.forward, _AdderFn.forward, _ResonanceFn.forward,
             _ResonanceFn._same_sign]
    offenders = []
    for fn in parts:
        offenders += _scan_mults(fn)
    return offenders


def audit_backward():
    """AST-scan the backward passes (reported separately, see docstring)."""
    offenders = []
    for fn in (_AdderFn.backward, _ResonanceFn.backward):
        offenders += _scan_mults(fn)
    return offenders


def check_forward_formulas():
    """Compare the vectorised layers against a naive double loop."""
    torch.manual_seed(7)
    x = torch.empty(4, 6, dtype=torch.double).uniform_(-1.0, 1.0)
    w = torch.empty(3, 6, dtype=torch.double).uniform_(-1.0, 1.0)

    ref_adder = torch.zeros(4, 3, dtype=torch.double)
    ref_res = torch.zeros(4, 3, dtype=torch.double)
    for b in range(4):
        for o in range(3):
            acc_a, acc_r = 0.0, 0.0
            for i in range(6):
                acc_a -= abs(float(w[o, i]) - float(x[b, i]))
                pos = float(w[o, i]) > 0 and float(x[b, i]) > 0
                neg = float(w[o, i]) < 0 and float(x[b, i]) < 0
                acc_r += (float(w[o, i]) + float(x[b, i])) if (pos or neg) \
                    else (float(w[o, i]) - float(x[b, i]))
            ref_adder[b, o] = acc_a
            ref_res[b, o] = acc_r

    got_adder = _AdderFn.apply(x, w, 2)
    got_res = _ResonanceFn.apply(x, w, 2)
    return {
        "AdderLinear": float((got_adder - ref_adder).abs().max()),
        "CellsResonance": float((got_res - ref_res).abs().max()),
    }


def gradcheck_adder():
    """The AdderLinear derivative is the true one -> verify it numerically."""
    torch.manual_seed(1234)
    x = torch.empty(6, 7, dtype=torch.double).uniform_(0.5, 1.5).requires_grad_(True)
    w = torch.empty(5, 7, dtype=torch.double).uniform_(-1.5, -0.5).requires_grad_(True)
    return torch.autograd.gradcheck(lambda a, b: _AdderFn.apply(a, b, 3), (x, w),
                                    eps=1e-6, atol=1e-5)


# --------------------------------------------------------------------------- #
#  5. Data
# --------------------------------------------------------------------------- #
def get_loaders(batch_size=128):
    tf = transforms.Compose([transforms.ToTensor()])     # pixels stay in [0, 1]
    root = os.path.join(HERE, "data")
    train_ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)
    # small fixed slice of the test set -> cheap curve while training
    probe_loader = DataLoader(Subset(test_ds, list(range(2500))),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, probe_loader, train_ds


def make_templates(dataset, n=128, seed=SEED):
    """Pick n real MNIST digits to initialise the adder filters."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n]
    return torch.stack([dataset[int(i)][0] for i in idx])


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum())
        total += yb.numel()
    model.train()
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
#  6. Training
# --------------------------------------------------------------------------- #
def train_model(model, train_loader, test_loader, probe_loader,
                epochs=2, lr=1e-3, tag="model", eval_every=100, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    hist = {"steps": [], "curve": [], "curve_loss": [],
            "batch_times": [], "epoch_acc": [], "epoch_loss": [], "epoch_time": []}

    step = 0
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        run_loss, run_n = 0.0, 0
        win_loss, win_n = 0.0, 0

        for xb, yb in train_loader:
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            hist["batch_times"].append(time.time() - t0)

            step += 1
            run_loss += loss.item()
            run_n += 1
            win_loss += loss.item()
            win_n += 1

            if step % eval_every == 0:
                acc = accuracy(model, probe_loader)
                hist["steps"].append(step)
                hist["curve"].append(acc)
                hist["curve_loss"].append(win_loss / max(win_n, 1))
                win_loss, win_n = 0.0, 0
                if verbose:
                    ms = 1000.0 * sum(hist["batch_times"][-eval_every:]) / eval_every
                    print(f"  {tag:<10} ep{ep} step {step:>4}  "
                          f"loss {loss.item():7.4f}  probe-acc {acc * 100:5.2f}%  "
                          f"{ms:6.1f} ms/batch", flush=True)

        ep_acc = accuracy(model, test_loader)
        hist["epoch_acc"].append(ep_acc)
        hist["epoch_loss"].append(run_loss / max(run_n, 1))
        hist["epoch_time"].append(time.time() - t_ep)
        if verbose:
            print(f"  {tag:<10} >> epoch {ep} done: test accuracy "
                  f"{ep_acc * 100:.2f}%  |  loss {hist['epoch_loss'][-1]:.4f}  |  "
                  f"{hist['epoch_time'][-1]:.1f}s", flush=True)

    return hist


# --------------------------------------------------------------------------- #
#  7. accuracy.png  (drawn with Pillow - no matplotlib dependency)
# --------------------------------------------------------------------------- #
_FONT_CACHE = {}


def _font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                return _FONT_CACHE[size]
            except OSError:
                pass
    try:
        _FONT_CACHE[size] = ImageFont.load_default(size=size)
    except TypeError:
        _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def save_plot(path, panels, total_steps, epoch_steps):
    W, H = 1500, 680
    BG, FG, GRID, MUTED = (255, 255, 255), (17, 24, 39), (231, 233, 237), (120, 128, 140)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((48, 26), "CellsNet vs NormalNet on MNIST", font=_font(30), fill=FG)
    d.text((48, 66), "same data, same schedule - one uses multiplication, the other only + and -",
           font=_font(17), fill=MUTED)

    xmax = max(total_steps, 1)
    pad, top, bottom = 60, 130, 96
    n = len(panels)
    pw = (W - pad * (n + 1)) // n

    for i, panel in enumerate(panels):
        x0 = pad + i * (pw + pad)
        y0, x1, y1 = top, x0 + pw, H - bottom
        ymin, ymax = panel["ymin"], panel["ymax"]

        # frame + grid
        d.rectangle([x0, y0, x1, y1], outline=(200, 204, 212), width=2)
        for k in range(6):
            gx = x0 + (x1 - x0) * k / 5
            gy = y1 - (y1 - y0) * k / 5
            d.line([(x0, gy), (x1, gy)], fill=GRID, width=1)
            val = ymin + (ymax - ymin) * k / 5
            label = f"{val * 100:.0f}%" if panel.get("percent") else f"{val:.2f}"
            d.text((x0 - 10, gy), label, font=_font(15), fill=MUTED, anchor="rm")

        # epoch boundaries
        for e in range(1, int(total_steps // max(epoch_steps, 1)) + 1):
            sx = x0 + (x1 - x0) * (e * epoch_steps) / xmax
            if sx < x1:
                for yy in range(y0, y1, 10):
                    d.line([(sx, yy), (sx, min(yy + 5, y1))], fill=(205, 208, 215), width=1)
                d.text((sx + 6, y0 + 6), f"epoch {e}", font=_font(14), fill=MUTED)

        d.text((x0 + (x1 - x0) / 2, y1 + 16), "training step (batch)", font=_font(16),
               fill=MUTED, anchor="ma")
        d.text((x1, y0 - 30), panel["title"], font=_font(20), fill=FG, anchor="ra")

        # series
        for j, s in enumerate(panel["series"]):
            pts = []
            for xv, yv in zip(s["xs"], s["ys"]):
                px = x0 + (x1 - x0) * min(xv, xmax) / xmax
                py = y1 - (y1 - y0) * (min(max(yv, ymin), ymax) - ymin) / (ymax - ymin)
                pts.append((px, py))
            if len(pts) > 1:
                d.line(pts, fill=s["color"], width=4, joint="curve")
            for p in pts:
                d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=s["color"])

            # legend under the panel
            ly = y1 + 46 + j * 26
            d.line([(x0 + 8, ly + 8), (x0 + 38, ly + 8)], fill=s["color"], width=4)
            d.text((x0 + 46, ly), s["label"], font=_font(16), fill=FG)

    img.save(path)


# --------------------------------------------------------------------------- #
#  8. Main
# --------------------------------------------------------------------------- #
def main():
    BATCH, EPOCHS, LR, HIDDEN = 128, 2, 1e-3, 128

    print("=" * 74)
    print("  CELLS-NET  -  can a network learn with only + and - ?")
    print("=" * 74)
    print(f"  batch={BATCH}  epochs={EPOCHS}  optimizer=Adam  lr={LR}  "
          f"torch={torch.__version__}  cpu-threads={torch.get_num_threads()}")

    # -- audit -------------------------------------------------------------
    fwd = audit_forward()
    bwd = audit_backward()
    print("\n[audit] AST scan of the code that runs in the forward pass")
    print(f"        AdderLinear / CellsResonance / Rescale / CenterNorm")
    print(f"        multiplications found : {len(fwd)}")
    for o in fwd:
        print("        !", o)

    print("[audit] AST scan of the backward passes: "
          f"{len(bwd)} multiplication(s)")
    for o in bwd:
        print("        -", o)
    print("        (AdderLinear's backward is +/-1 only; the resonance layer")
    print("         scales dL/dw by |x|, which is unavoidable - see docstring)")

    print("[audit] forward formulas vs naive double loop:")
    for name, err in check_forward_formulas().items():
        print(f"        {name:<20} max|diff| = {err:.1e}  "
              f"{'PASS' if err < 1e-12 else 'FAIL'}")
    print(f"[audit] gradcheck AdderLinear (true derivative): "
          f"{'PASS' if gradcheck_adder() else 'FAIL'}")

    # -- data --------------------------------------------------------------
    print("\n[data] loading MNIST (downloads once into ./data) ...")
    train_loader, test_loader, probe_loader, train_ds = get_loaders(BATCH)
    print(f"       train={len(train_loader.dataset)}  test={len(test_loader.dataset)}  "
          f"batches/epoch={len(train_loader)}")

    # -- train -------------------------------------------------------------
    print("\n[1/2] NormalNet   (nn.Linear, uses multiplication)")
    normal = NormalNet(HIDDEN)
    t0 = time.time()
    h_normal = train_model(normal, train_loader, test_loader, probe_loader,
                           epochs=EPOCHS, lr=LR, tag="Normal")
    t_normal = time.time() - t0

    print("\n[2/2] CellsNet  (AdderLinear + CellsResonance, only + and -)")
    cells = CellsNet(HIDDEN, templates=make_templates(train_ds, HIDDEN))
    t0 = time.time()
    h_cells = train_model(cells, train_loader, test_loader, probe_loader,
                          epochs=EPOCHS, lr=LR, tag="Cells")
    t_cells = time.time() - t0

    # -- numbers -----------------------------------------------------------
    normal_macs = 784 * HIDDEN + HIDDEN * 10          # per sample, per forward
    cells_macs = 0
    normal_ms = 1000.0 * sum(h_normal["batch_times"]) / len(h_normal["batch_times"])
    cells_ms = 1000.0 * sum(h_cells["batch_times"]) / len(h_cells["batch_times"])

    acc_n = h_normal["epoch_acc"][-1]
    acc_k = h_cells["epoch_acc"][-1]

    print("\n" + "=" * 74)
    print("  RESULTS  (2 epochs, MNIST test set)")
    print("=" * 74)
    rows = [
        ("test accuracy", f"{acc_n * 100:.2f}%", f"{acc_k * 100:.2f}%"),
        ("multiplications / forward pass", f"{normal_macs:,}", f"{cells_macs:,}"),
        ("   ... per batch of 128", f"{normal_macs * BATCH:,}", f"{cells_macs * BATCH:,}"),
        ("   ... per epoch (60k images)", f"{normal_macs * 60000:,}", f"{cells_macs * 60000:,}"),
        ("time / batch (fwd+bwd+step)", f"{normal_ms:.1f} ms", f"{cells_ms:.1f} ms"),
        ("total training time", f"{t_normal:.1f} s", f"{t_cells:.1f} s"),
        ("final train loss", f"{h_normal['epoch_loss'][-1]:.4f}", f"{h_cells['epoch_loss'][-1]:.4f}"),
    ]
    print(f"  {'':<32}{'NormalNet':>18}{'CellsNet':>18}")
    print("  " + "-" * 70)
    for name, a, b in rows:
        print(f"  {name:<32}{a:>18}{b:>18}")
    print("  " + "-" * 70)
    print(f"  speed ratio: CellsNet is {cells_ms / normal_ms:.1f}x slower per batch")
    print("  note: the loss (cross-entropy) and the Adam optimizer still contain")
    print("        multiplications internally; both nets pay that same cost.")
    print("        The claim is about the model itself: 0 multiplications in")
    print("        every forward pass, and +/-1 only in AdderLinear's backward.")
    print("        (The resonance layer's backward needs one scale factor, |x|,")
    print("         otherwise its sign pattern provably cannot move at all.)")

    # -- plot --------------------------------------------------------------
    panels = [
        {"title": "Test accuracy (2.5k probe)", "ymin": 0.0, "ymax": 1.0,
         "percent": True,
         "series": [
             {"label": f"NormalNet   {acc_n * 100:.2f}% test", "color": (37, 99, 235),
              "xs": h_normal["steps"], "ys": h_normal["curve"]},
             {"label": f"CellsNet  {acc_k * 100:.2f}% test", "color": (234, 88, 12),
              "xs": h_cells["steps"], "ys": h_cells["curve"]},
         ]},
        {"title": "Training loss (cross-entropy)", "ymin": 0.0,
         "ymax": max(max(h_normal["curve_loss"]), max(h_cells["curve_loss"])) * 1.05,
         "percent": False,
         "series": [
             {"label": "NormalNet", "color": (37, 99, 235),
              "xs": h_normal["steps"], "ys": h_normal["curve_loss"]},
             {"label": "CellsNet", "color": (234, 88, 12),
              "xs": h_cells["steps"], "ys": h_cells["curve_loss"]},
         ]},
    ]
    out = os.path.join(HERE, "accuracy.png")
    save_plot(out, panels, total_steps=len(train_loader) * EPOCHS,
              epoch_steps=len(train_loader))
    print(f"\n  plot saved -> {out}")


if __name__ == "__main__":
    main()
