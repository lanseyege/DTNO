"""
Low-cost sanity baselines (§3).

Persistence   U_hat(t+h) = U(t).  The floor.  Any horizon where a trained model
              loses to persistence is a horizon where it has learned nothing.

POD-DMD       U(t) -> a_t (POD coefficients), a_{t+h} = A^h a_t.  This one earns
              its place: it is itself a *direct finite-time* evolution, so it
              tests the formulation and not just the network.  A^h is evaluated
              through the eigendecomposition, so its cost is independent of h --
              exactly the constant-depth property we are claiming for the neural
              operator, on a model with no learned nonlinearity at all.  If
              DT-FNO cannot beat POD-DMD, the problem is the formulation or the
              data pipeline, not the architecture (§3).

Both operate in NORMALISED space, using the same frozen Normalizer as the neural
models, so every number in §25 is computed in the same units.

Implementation notes for POD:
  * The time-mean is subtracted first.  A swirl burner has a strong standing
    mean field; POD on the raw field spends its leading modes describing the
    mean and the fluctuation dynamics never get resolved.
  * Modes come from the method of snapshots (Gram matrix, n_snap x n_snap)
    rather than an SVD of the 245,760-column snapshot matrix.
  * Eigenvalue magnitudes of A are clipped to <= 1 by default.  Least-squares
    DMD routinely returns |lambda| = 1.001, invisible at h = 8 and a factor of
    1.7 at h = 512, which would turn the baseline into a divergence artefact
    rather than a baseline.  Set clip_eigs=False to see the raw behaviour.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np
import torch


class Persistence:
    """U_hat(t + h) = U(t) for all h."""

    name = "persistence"

    def predict(self, x: torch.Tensor, h: int, **kw) -> torch.Tensor:
        """x: (B, K, H, W, C) -> (B, H, W, C)."""
        return x[:, -1]

    @staticmethod
    def n_model_evals(h: int) -> int:
        return 0

    def summary(self) -> str:
        return "Persistence | params=0"


class TrajectoryClimatology:
    """ORACLE: predict the time-mean field of the source trajectory itself.

    Not a forecasting method -- it peeks at the test trajectory's own
    statistics. It exists to answer one question that the error curve alone
    cannot: once E(h) goes flat, has the operator converged to CONDITIONAL
    CLIMATOLOGY?

    Beyond the deterministic predictability horizon the minimum-MSE prediction
    is the conditional mean, which for a statistically stationary trajectory is
    its own time average -- a single field, independent of h. A model that has
    reached that point has stopped doing dynamics and started doing "identify
    the operating point, emit its mean field". That is a real capability and
    worth reporting, but it is NOT a finite-time flow map, and the difference
    decides how §48's scientific claim may be worded.

    Reading it:
        E_model(h) >> E_clim   the model still carries dynamical information
        E_model(h) ~= E_clim   the model has converged to conditional climatology
        E_model(h) <  E_clim   genuine dynamics beyond what the mean field gives

    Because it is an oracle, it is a LOWER bound on the error of any
    no-dynamics predictor, not a baseline anyone could deploy. Label it as such
    in every figure.
    """

    name = "climatology"

    def __init__(self):
        self.means: Dict[int, np.ndarray] = {}
        self._traj = None

    def fit(self, store, traj_indices, channel_indices, normalizer,
            t_stride: int = 10, verbose: bool = True):
        for t in traj_indices:
            idx = list(range(0, store.T, t_stride))
            raw = store.read(int(t), idx, channel_indices)
            self.means[int(t)] = normalizer.forward(raw).mean(axis=0)
            if verbose:
                print(f"    traj {t:>3}: climatology over {len(idx)} frames")
        return self

    def set_context(self, traj):
        self._traj = traj

    def predict(self, x: torch.Tensor, h: int, **kw) -> torch.Tensor:
        B = x.shape[0]
        if self._traj is None:
            raise RuntimeError("call set_context(traj) before predict")
        out = np.stack([self.means[int(t)] for t in self._traj[:B]], axis=0)
        return torch.from_numpy(out).to(x.device, x.dtype)

    @staticmethod
    def n_model_evals(h: int) -> int:
        return 0

    def summary(self) -> str:
        return (f"TrajectoryClimatology (ORACLE) | {len(self.means)} "
                f"trajectory means | params=0")


class NearestTrainClimatology:
    """Deployable no-dynamics baseline: the climatology of the most similar
    TRAINING trajectory.

    Why this exists. `TrajectoryClimatology` is an oracle -- it reads the test
    trajectory's own time-mean, which no forecaster has. Comparing a model
    against it answers "is the model as good as knowing the answer", which is
    not a question anyone asked, and reporting a model as 'worse than
    climatology' on that basis overstates the case.

    This baseline uses training data only. It averages the K history frames the
    model itself receives, matches that against each training trajectory's mean
    field, and returns the closest one (or a softmax-weighted blend). Fully
    deployable, contains no dynamics, and its output does not depend on h.

    The three numbers together are what should be reported:

        E_oracle    (TrajectoryClimatology)  what knowing the answer buys
        E_nearest   (this)                   what a real no-dynamics method buys
        E_model                              the operator under test

    E_model < E_nearest is the claim worth making. The gap between E_nearest and
    E_oracle measures how much of the operating point is actually inferable from
    K frames and a 20-trajectory training set -- which is a property of the
    dataset, not of the model.
    """

    name = "nearest_climatology"

    def __init__(self, k: int = 1, temperature: float = 0.0):
        self.k = int(k)
        # temperature > 0 blends the k neighbours with softmax(-d^2 / T)
        # weights, which estimates a conditional mean rather than committing to
        # one trajectory. 0 = hard nearest neighbour.
        self.temperature = float(temperature)
        self.means_: Optional[np.ndarray] = None      # (n_train, H, W, C)
        self.traj_ids_: List[int] = []
        self._flat: Optional[np.ndarray] = None

    def fit(self, store, train_traj_indices, channel_indices, normalizer,
            t_stride: int = 10, verbose: bool = True):
        mus = []
        for t in train_traj_indices:
            idx = list(range(0, store.T, t_stride))
            raw = store.read(int(t), idx, channel_indices)
            mus.append(normalizer.forward(raw).mean(axis=0))
            self.traj_ids_.append(int(t))
        self.means_ = np.stack(mus, axis=0).astype(np.float32)
        self._flat = self.means_.reshape(len(mus), -1)
        if verbose:
            print(f"    fitted on {len(mus)} TRAINING trajectories "
                  f"{self.traj_ids_}")
        return self

    def predict(self, x: torch.Tensor, h: int, **kw) -> torch.Tensor:
        """x: (B, K, H, W, C) -> (B, H, W, C), independent of h."""
        q = x.detach().float().mean(dim=1).cpu().numpy()      # (B, H, W, C)
        B = q.shape[0]
        qf = q.reshape(B, -1)
        # squared distances to every training climatology
        d2 = ((qf[:, None, :] - self._flat[None, :, :]) ** 2).sum(axis=-1)
        order = np.argsort(d2, axis=1)[:, : self.k]
        if self.temperature > 0:
            out = np.zeros_like(q)
            for b in range(B):
                sel = order[b]
                w = np.exp(-d2[b, sel] / (self.temperature * d2[b, sel].min()
                                          + 1e-30))
                w = w / w.sum()
                out[b] = np.tensordot(w, self.means_[sel], axes=(0, 0))
        else:
            out = self.means_[order].mean(axis=1)
        return torch.from_numpy(out.astype(np.float32)).to(x.device, x.dtype)

    @staticmethod
    def n_model_evals(h: int) -> int:
        return 0

    def summary(self) -> str:
        n = 0 if self.means_ is None else self.means_.shape[0]
        mode = ("nearest" if self.temperature <= 0
                else f"softmax blend T={self.temperature}")
        return f"NearestTrainClimatology | {n} training means | k={self.k} | {mode}"


class PODDMD:
    """POD projection + linear DMD propagator in coefficient space."""

    name = "pod_dmd"

    def __init__(self, rank: int = 128, subtract_mean: bool = True,
                 clip_eigs: bool = True, ridge: float = 1e-6):
        self.rank = int(rank)
        self.subtract_mean = bool(subtract_mean)
        self.clip_eigs = bool(clip_eigs)
        self.ridge = float(ridge)
        self.mean_: Optional[np.ndarray] = None       # (F,)
        self.modes_: Optional[np.ndarray] = None      # (F, r)
        self.A_: Optional[np.ndarray] = None          # (r, r)
        self.eigvals_: Optional[np.ndarray] = None
        self.eigvecs_: Optional[np.ndarray] = None
        self.eigvecs_inv_: Optional[np.ndarray] = None
        self.shape_: Optional[tuple] = None           # (H, W, C)

    # -- fit --------------------------------------------------------------
    def fit(self, snapshots: Sequence[np.ndarray], verbose: bool = True):
        """snapshots: list of (T_i, H, W, C) NORMALISED trajectory arrays."""
        H, W, C = snapshots[0].shape[1:]
        self.shape_ = (H, W, C)
        F = H * W * C

        X = np.concatenate([s.reshape(s.shape[0], F) for s in snapshots], axis=0)
        X = X.astype(np.float32, copy=False)
        n = X.shape[0]
        if verbose:
            print(f"  POD: {n} snapshots x {F} features "
                  f"({X.nbytes / 1e9:.2f} GB)")

        if self.subtract_mean:
            self.mean_ = X.mean(axis=0)
            X = X - self.mean_
        else:
            self.mean_ = np.zeros(F, dtype=np.float32)

        r = min(self.rank, n - 1)
        G = (X @ X.T).astype(np.float64)              # (n, n) method of snapshots
        w, V = np.linalg.eigh(G)
        order = np.argsort(w)[::-1][:r]
        w, V = np.clip(w[order], 1e-12, None), V[:, order]
        modes = (X.T @ V) / np.sqrt(w)[None, :]       # (F, r), orthonormal
        self.modes_ = modes.astype(np.float32)
        energy = float(w.sum() / max(np.linalg.eigvalsh(G).sum(), 1e-12))
        if verbose:
            print(f"  POD: rank {r}, captured energy {100 * energy:.2f}%")

        # -- DMD on per-trajectory coefficient sequences ------------------
        A0, A1 = [], []
        for s in snapshots:
            a = (s.reshape(s.shape[0], F) - self.mean_) @ self.modes_   # (T_i, r)
            A0.append(a[:-1])
            A1.append(a[1:])
        A0 = np.concatenate(A0, axis=0).astype(np.float64)
        A1 = np.concatenate(A1, axis=0).astype(np.float64)

        gram = A0.T @ A0 + self.ridge * np.trace(A0.T @ A0) / r * np.eye(r)
        self.A_ = np.linalg.solve(gram, A0.T @ A1).T                    # (r, r)

        lam, vec = np.linalg.eig(self.A_)
        if self.clip_eigs:
            mag = np.abs(lam)
            over = mag > 1.0
            if over.any():
                lam = np.where(over, lam / mag, lam)
                if verbose:
                    print(f"  DMD: clipped {int(over.sum())}/{r} eigenvalues to "
                          f"|lambda| = 1 (max was {mag.max():.4f})")
        self.eigvals_, self.eigvecs_ = lam, vec
        self.eigvecs_inv_ = np.linalg.pinv(vec)
        if verbose:
            print(f"  DMD: |lambda| in [{np.abs(lam).min():.4f}, "
                  f"{np.abs(lam).max():.4f}]")
        return self

    # -- predict ----------------------------------------------------------
    def _propagate(self, a: np.ndarray, h: int) -> np.ndarray:
        """a: (B, r) -> A^h a, via the eigendecomposition (cost is h-free)."""
        lam_h = self.eigvals_ ** float(h)
        M = (self.eigvecs_ * lam_h[None, :]) @ self.eigvecs_inv_
        return np.real(a.astype(np.complex128) @ M.T)

    def predict(self, x: torch.Tensor, h: int, **kw) -> torch.Tensor:
        """x: (B, K, H, W, C) normalised -> (B, H, W, C)."""
        dev, dt = x.device, x.dtype
        u = x[:, -1].detach().float().cpu().numpy()
        B = u.shape[0]
        F = int(np.prod(self.shape_))
        a = (u.reshape(B, F) - self.mean_) @ self.modes_
        a_h = self._propagate(a, h)
        out = a_h @ self.modes_.T + self.mean_
        return torch.from_numpy(out.reshape(B, *self.shape_).astype(np.float32)).to(dev, dt)

    @staticmethod
    def n_model_evals(h: int) -> int:
        return 1                      # one direct finite-time application

    def summary(self) -> str:
        r = 0 if self.modes_ is None else self.modes_.shape[1]
        return f"POD-DMD | rank={r} | mean_subtracted={self.subtract_mean}"

    # -- io ---------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path, mean=self.mean_, modes=self.modes_, A=self.A_,
            shape=np.array(self.shape_),
            meta=np.array(json.dumps({
                "rank": self.rank, "subtract_mean": self.subtract_mean,
                "clip_eigs": self.clip_eigs, "ridge": self.ridge})))

    @classmethod
    def load(cls, path: str) -> "PODDMD":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        obj = cls(**meta)
        obj.mean_, obj.modes_, obj.A_ = z["mean"], z["modes"], z["A"]
        obj.shape_ = tuple(int(v) for v in z["shape"])
        lam, vec = np.linalg.eig(obj.A_)
        if obj.clip_eigs:
            mag = np.abs(lam)
            lam = np.where(mag > 1.0, lam / mag, lam)
        obj.eigvals_, obj.eigvecs_ = lam, vec
        obj.eigvecs_inv_ = np.linalg.pinv(vec)
        return obj
