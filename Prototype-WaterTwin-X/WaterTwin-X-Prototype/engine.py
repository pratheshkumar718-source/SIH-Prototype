"""
WaterTwin-X prototype - core engine (multi-parameter).

Same three signals as before (uncertainty, surprise, disagreement) plus a
static hazard weighting - but now tracking THREE water-quality parameters
at once, not just dissolved oxygen:

  - Dissolved oxygen (DO)  - the "fish die-off" signal, anchor parameter,
    carries the full mechanism: uncertainty + surprise + hypothesis
    discrimination (point-source vs. diffuse).
  - pH                     - a second, independent problem (e.g. an acidic
    industrial discharge), tracked with uncertainty + hazard only, to keep
    the demo legible - it deliberately does NOT duplicate the full DO
    mechanism, it shows the architecture generalizes to other parameters.
  - Turbidity               - a third, independent problem (e.g. runoff/silt),
    same lighter treatment as pH.

A single sensor probe reads all three parameters at once when deployed - that's
realistic (a real water-quality sonde does exactly this) - so all three share
the same `known_idx` list.

The combined priority score at each cell is the WORST (max) of the three
parameters' individual urgency scores - one badly-behaving parameter is
enough to justify sending a sensor there, even if the other two look fine.

No AI/LLM here on purpose - this is spatial regression + model comparison.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

GRID_N = 15

SURPRISE_DECAY = 0.8
SURPRISE_RADIUS = 0.18
SURPRISE_Z_THRESHOLD = 2.0   # raised from 1.0 after real-data validation showed the looser
                              # threshold caused the mechanism to chase ordinary measurement
                              # noise instead of genuine anomalies - see real_data_report.md
HYPOTHESIS_TEMPERATURE = 6.0
EPSILON_EXPLORE = 0.15  # fraction of "priority" steps spent on forced random
                         # exploration - guarantees no real anomaly can stay
                         # permanently buried just because early samples
                         # happened not to land near it (a real failure mode
                         # found during testing - pure score-following can
                         # get unlucky and never discover a distant problem)

# per-parameter configuration. "scale" is the characteristic size of a real
# anomaly for that parameter - used only to normalize errors fairly when
# combining three very different units into one comparison metric.
PARAMS = {
    "do": {
        "label": "Dissolved Oxygen", "unit": "mg/L", "baseline": 7.0, "noise": 0.15,
        "hotspots": [(0.78, 0.25, 0.16, -6.0)],   # (x, y, radius, depth) - depth negative = dip
        "alert_low": 3.0, "alert_high": None, "scale": 6.0,
        "primary": True,
    },
    "ph": {
        "label": "pH", "unit": "", "baseline": 7.2, "noise": 0.10,
        "hotspots": [(0.22, 0.72, 0.13, -2.6)],   # an acidic-discharge dip, different location
        "alert_low": 5.5, "alert_high": 8.5, "scale": 2.6,
        "primary": False,
    },
    "turbidity": {
        "label": "Turbidity", "unit": "NTU", "baseline": 5.0, "noise": 1.0,
        "hotspots": [(0.5, 0.85, 0.15, 22.0)],    # a silt/runoff spike, third location
        "alert_low": None, "alert_high": 25.0, "scale": 22.0,
        "primary": False,
    },
}

W_UNCERTAINTY = 0.35
W_HAZARD = 0.25
W_SURPRISE = 0.20
W_DISAGREEMENT = 0.20
W_UNCERTAINTY_SECONDARY = 0.30
W_HAZARD_SECONDARY = 0.20
W_SURPRISE_SECONDARY = 0.50


def _normalize(a):
    a = a.copy()
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 1e-9 else np.zeros_like(a)


def make_field(seed, cfg):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 1, GRID_N), np.linspace(0, 1, GRID_N))
    field = cfg["baseline"] + rng.normal(0, cfg["noise"], size=xx.shape)
    for (hx, hy, hr, depth) in cfg["hotspots"]:
        dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        field += depth * np.exp(-(dist ** 2) / (2 * hr ** 2))
    return field


def make_hazard_map():
    xx, yy = np.meshgrid(np.linspace(0, 1, GRID_N), np.linspace(0, 1, GRID_N))
    settlement_x, settlement_y = 0.9, 0.1
    dist = np.sqrt((xx - settlement_x) ** 2 + (yy - settlement_y) ** 2)
    hazard = np.exp(-(dist ** 2) / (2 * 0.55 ** 2))
    return hazard / hazard.max()


class WaterTwin:
    def __init__(self, seed=7, n_initial=7):
        self.xx, self.yy = np.meshgrid(np.linspace(0, 1, GRID_N), np.linspace(0, 1, GRID_N))
        self.coords = np.column_stack([self.xx.ravel(), self.yy.ravel()])
        self.hazard_map = make_hazard_map()

        self.true_fields = {name: make_field(seed + i, cfg) for i, (name, cfg) in enumerate(PARAMS.items())}

        rng = np.random.default_rng(seed + 100)
        all_idx = np.arange(GRID_N * GRID_N)
        self.known_idx = list(rng.choice(all_idx, size=n_initial, replace=False))

        self.surprise_maps = {name: np.zeros((GRID_N, GRID_N)) for name in PARAMS}
        self.last_surprise = None
        self.history = []

        self.pred = {}         # name -> field
        self.uncertainty = {}  # name -> field
        self._fit()

    # ---------- fitting ----------
    def _fit_gp(self, true_field, length_scale, optimize):
        X_known = self.coords[self.known_idx]
        y_known = true_field.ravel()[self.known_idx]
        if optimize:
            kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=length_scale, length_scale_bounds=(0.05, 1.0)) \
                + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-3, 1.0))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
        else:
            kernel = C(1.0, "fixed") * RBF(length_scale=length_scale, length_scale_bounds="fixed") \
                + WhiteKernel(noise_level=0.05, noise_level_bounds="fixed")
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)
        gp.fit(X_known, y_known)
        mean, std = gp.predict(self.coords, return_std=True)
        return gp, mean.reshape(GRID_N, GRID_N), std.reshape(GRID_N, GRID_N)

    def _fit(self):
        for name, cfg in PARAMS.items():
            gp, mean, std = self._fit_gp(self.true_fields[name], length_scale=0.3, optimize=True)
            self.pred[name] = mean
            self.uncertainty[name] = std

        # hypothesis discrimination + surprise map only live on the primary (DO) parameter
        do_true = self.true_fields["do"]
        self.gp_h1, self.h1_field, _ = self._fit_gp(do_true, length_scale=0.12, optimize=False)
        self.gp_h2, self.h2_field, _ = self._fit_gp(do_true, length_scale=0.55, optimize=False)
        self.disagreement = np.abs(self.h1_field - self.h2_field)

        ll1 = self.gp_h1.log_marginal_likelihood(self.gp_h1.kernel_.theta) / HYPOTHESIS_TEMPERATURE
        ll2 = self.gp_h2.log_marginal_likelihood(self.gp_h2.kernel_.theta) / HYPOTHESIS_TEMPERATURE
        m = max(ll1, ll2)
        e1, e2 = np.exp(ll1 - m), np.exp(ll2 - m)
        self.hypothesis_confidence = {"point_source": float(e1 / (e1 + e2)), "diffuse": float(e2 / (e1 + e2))}

    # ---------- scoring ----------
    def _param_score(self, name):
        cfg = PARAMS[name]
        u = _normalize(self.uncertainty[name])
        h = self.hazard_map
        s = _normalize(self.surprise_maps[name])
        if cfg["primary"]:
            d = _normalize(self.disagreement)
            return W_UNCERTAINTY * u + W_HAZARD * h + W_SURPRISE * s + W_DISAGREEMENT * d
        # secondary parameters: no hypothesis-discrimination model, but they DO
        # get their own surprise tracking - without it they just chase whatever
        # corner of the grid is least-sampled, ignoring their own real anomaly
        return W_UNCERTAINTY_SECONDARY * u + W_HAZARD_SECONDARY * h + W_SURPRISE_SECONDARY * s

    def score_map(self):
        per_param = [self._param_score(name) for name in PARAMS]
        combined = np.maximum.reduce(per_param)  # worst parameter governs
        flat = combined.ravel().copy()
        flat[self.known_idx] = -1
        return combined, flat

    def best_next_index(self):
        _, flat_score = self.score_map()
        if np.all(flat_score < 0):
            return None
        return int(np.argmax(flat_score))

    def random_next_index(self, rng=None):
        rng = rng or np.random.default_rng()
        remaining = [i for i in range(GRID_N * GRID_N) if i not in self.known_idx]
        if not remaining:
            return None
        return int(rng.choice(remaining))

    # ---------- actions ----------
    def _apply_surprise(self, name, idx, z_score):
        smap = self.surprise_maps[name]
        smap *= SURPRISE_DECAY
        if abs(z_score) >= SURPRISE_Z_THRESHOLD:
            r, c = divmod(idx, GRID_N)
            cx, cy = self.xx[r, c], self.yy[r, c]
            dist2 = (self.xx - cx) ** 2 + (self.yy - cy) ** 2
            bump = min(abs(z_score) / 4.0, 1.0) * np.exp(-dist2 / (2 * SURPRISE_RADIUS ** 2))
            self.surprise_maps[name] = np.maximum(smap, bump)
        else:
            self.surprise_maps[name] = smap

    def step(self, method="priority"):
        if method == "priority" and np.random.default_rng().random() < EPSILON_EXPLORE:
            idx = self.random_next_index()  # forced exploration, still counted as a "priority" step
        else:
            idx = self.best_next_index() if method == "priority" else self.random_next_index()
        if idx is None:
            return None

        r, c = divmod(idx, GRID_N)

        # compute a surprise z-score for EVERY parameter at this cell, using
        # each parameter's own pre-reveal prediction/uncertainty - a
        # discharge could show up as a DO problem, a pH problem, or both
        z_scores = {}
        for name in PARAMS:
            pred_before = float(self.pred[name][r, c])
            std_before = max(float(self.uncertainty[name][r, c]), 1e-6)
            true_val = float(self.true_fields[name][r, c])
            z = (true_val - pred_before) / std_before
            z_scores[name] = z
            self._apply_surprise(name, idx, z)

        # report whichever parameter was surprised the most, so the banner
        # generalizes across all three instead of only ever talking about DO
        loudest = max(z_scores, key=lambda n: abs(z_scores[n]))
        is_surprising = abs(z_scores[loudest]) >= SURPRISE_Z_THRESHOLD
        self.last_surprise = {
            "parameter": loudest, "row": r, "col": c,
            "predicted": round(float(self.pred[loudest][r, c]), 3),
            "actual": round(float(self.true_fields[loudest][r, c]), 3),
            "z_score": round(z_scores[loudest], 2), "surprising": bool(is_surprising)
        } if (method == "priority" and is_surprising) else None

        self.known_idx.append(idx)
        self._fit()

        readings = {name: round(float(self.true_fields[name][r, c]), 3) for name in PARAMS}
        self.history.append({
            "step": len(self.history) + 1, "chosen_idx": idx, "method": method,
            "readings": readings, "z_scores": {k: round(v, 2) for k, v in z_scores.items()},
            "surprising": bool(is_surprising)
        })
        return idx

    def manual_edit(self, row, col, field, new_value):
        if field not in PARAMS:
            raise ValueError(f"unknown field '{field}', must be one of {list(PARAMS)}")
        if not (0 <= row < GRID_N and 0 <= col < GRID_N):
            raise ValueError(f"row/col must be within 0..{GRID_N-1}")
        self.true_fields[field][row, col] = new_value
        self._fit()

    # ---------- metrics ----------
    def total_error(self):
        """Combined error across all three parameters, each normalized by its
        own characteristic scale so DO/pH/turbidity contribute fairly despite
        very different units."""
        errs = []
        for name, cfg in PARAMS.items():
            raw = float(np.mean(np.abs(self.pred[name] - self.true_fields[name])))
            errs.append(raw / cfg["scale"])
        return float(np.mean(errs))

    def alerts(self):
        out = []
        for name, cfg in PARAMS.items():
            field = self.pred[name]
            if cfg["alert_low"] is not None:
                for r, c in np.argwhere(field < cfg["alert_low"]):
                    out.append({"parameter": name, "row": int(r), "col": int(c),
                                "value": round(float(field[r, c]), 3), "direction": "low"})
            if cfg["alert_high"] is not None:
                for r, c in np.argwhere(field > cfg["alert_high"]):
                    out.append({"parameter": name, "row": int(r), "col": int(c),
                                "value": round(float(field[r, c]), 3), "direction": "high"})
        return out

    # ---------- serialization ----------
    def state(self):
        score_grid, _ = self.score_map()
        known_mask = np.zeros(GRID_N * GRID_N, dtype=bool)
        known_mask[self.known_idx] = True
        known_mask = known_mask.reshape(GRID_N, GRID_N)

        parameters = {}
        for name, cfg in PARAMS.items():
            parameters[name] = {
                "label": cfg["label"], "unit": cfg["unit"],
                "predicted": self.pred[name].round(3).tolist(),
                "uncertainty": self.uncertainty[name].round(3).tolist(),
                "alert_low": cfg["alert_low"], "alert_high": cfg["alert_high"],
                "scale": cfg["scale"],
            }

        return {
            "grid_n": GRID_N,
            "parameters": parameters,
            "hazard": self.hazard_map.round(3).tolist(),
            "surprise": {name: m.round(3).tolist() for name, m in self.surprise_maps.items()},
            "disagreement": self.disagreement.round(3).tolist(),
            "score": score_grid.round(3).tolist(),
            "known_mask": known_mask.tolist(),
            "alerts": self.alerts(),
            "total_error": round(self.total_error(), 4),
            "steps_taken": len(self.history),
            "history": self.history,
            "hypothesis_confidence": self.hypothesis_confidence,
            "last_surprise": self.last_surprise,
        }
