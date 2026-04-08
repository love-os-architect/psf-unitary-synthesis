from __future__ import annotations

import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import Protocol, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, RZZGate
from qiskit.transpiler.passes.synthesis.plugin import UnitarySynthesisPlugin

# ============================================================
# Math primitives
# ============================================================

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.array([[c, s], [s, c]], dtype=complex)

def kron(*ops: np.ndarray) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def Uzz(tau: float) -> np.ndarray:
    ph = np.exp(-1j * 0.5 * tau)
    phc = np.conjugate(ph)
    return np.diag([ph, phc, phc, ph])

# ============================================================
# Fidelity metric (metric != loss)
# ============================================================

def average_gate_fidelity(U: np.ndarray, V: np.ndarray) -> float:
    d = U.shape[0]
    tr = np.trace(U.conj().T @ V)
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))

def infidelity(U: np.ndarray, V: np.ndarray) -> float:
    return 1.0 - average_gate_fidelity(U, V)

# ============================================================
# Regularization policies
# ============================================================

class Regularizer(Protocol):
    def grad(self, x: np.ndarray) -> np.ndarray: ...

class BoundedProjectiveRegularizer:
    """
    Bounded projective regularization:
      x -> x / sqrt(1 + x^2)

    Softly constrains parameters without hard clipping.
    """
    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x / (1.0 + x**2)**2

class SparsitySmoothnessRegularizer:
    """
    Composite L1 + total-variation prior
    (sparsity and hardware-friendly smooth pulses).
    """
    def __init__(self, beta_l1: float, beta_tv: float, shape_angles, shape_taus):
        self.beta_l1 = beta_l1
        self.beta_tv = beta_tv
        self.shape_angles = shape_angles
        self.shape_taus = shape_taus

    def grad(self, x: np.ndarray) -> np.ndarray:
        na = np.prod(self.shape_angles)
        g = np.zeros_like(x)

        angles = x[:na].reshape(self.shape_angles)
        taus   = x[na:].reshape(self.shape_taus)

        g_angles = np.sign(angles)
        g_taus   = np.sign(taus)

        # TV penalties
        diff_a = np.diff(angles, axis=0)
        tv_a = np.zeros_like(angles)
        s = np.sign(diff_a)
        tv_a[0]     -= s[0]
        tv_a[-1]    += s[-1]
        if angles.shape[0] > 2:
            tv_a[1:-1] += s[:-1] - s[1:]

        if taus.size >= 2:
            diff_t = np.diff(taus)
            tv_t = np.zeros_like(taus)
            s2 = np.sign(diff_t)
            tv_t[0]     -= s2[0]
            tv_t[-1]    += s2[-1]
            if taus.size > 2:
                tv_t[1:-1] += s2[:-1] - s2[1:]
        else:
            tv_t = np.zeros_like(taus)

        g[:na] = self.beta_l1 * g_angles.ravel() + self.beta_tv * tv_a.ravel()
        g[na:] = self.beta_l1 * g_taus.ravel()   + self.beta_tv * tv_t.ravel()

        return g

# ============================================================
# Optimizer
# ============================================================

@dataclass
class AdamConfig:
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_step_norm: float = 1.5

class AdamOptimizer:
    def __init__(self, dim: int, cfg: AdamConfig):
        self.cfg = cfg
        self.m1 = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        b1, b2 = self.cfg.betas

        self.m1 = b1 * self.m1 + (1 - b1) * grad
        self.m2 = b2 * self.m2 + (1 - b2) * (grad ** 2)

        m1h = self.m1 / (1 - b1 ** self.t)
        m2h = self.m2 / (1 - b2 ** self.t)

        step = self.cfg.lr * m1h / (np.sqrt(m2h) + self.cfg.eps)
        n = np.linalg.norm(step)
        if n > self.cfg.max_step_norm:
            step *= self.cfg.max_step_norm / (n + 1e-12)
        return step

# ============================================================
# Circuit model
# ============================================================

class TwoQubitRZZModel:
    def __init__(self, depth: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.depth = depth
        self.angles = rng.normal(scale=0.2, size=(depth + 1, 2, 3))
        self.taus   = rng.normal(scale=0.2, size=(depth,))

    def flat(self) -> np.ndarray:
        return np.concatenate([self.angles.ravel(), self.taus.ravel()])

    def set_flat(self, v: np.ndarray):
        na = self.angles.size
        self.angles = v[:na].reshape(self.angles.shape)
        self.taus   = v[na:].reshape(self.taus.shape)

    def unitary(self) -> np.ndarray:
        U = self._local_block(self.angles[0])
        for k in range(self.depth):
            U = Uzz(self.taus[k]) @ U
            U = self._local_block(self.angles[k + 1]) @ U
        return U

    def _local_block(self, a) -> np.ndarray:
        return kron(
            Rz(a[0, 2]) @ Ry(a[0, 1]) @ Rx(a[0, 0]),
            Rz(a[1, 2]) @ Ry(a[1, 1]) @ Rx(a[1, 0])
        )

# ============================================================
# Trainer
# ============================================================

class NumericalSynthesizer:
    """
    Hardware-agnostic numerical synthesis with
    physically motivated regularization policies.
    """
    PS = np.pi / 2

    def __init__(
        self,
        model: TwoQubitRZZModel,
        optimizer: AdamOptimizer,
        regularizers: List[Regularizer]
    ):
        self.model = model
        self.opt = optimizer
        self.regularizers = regularizers

    def step(self, target: np.ndarray) -> float:
        base = self.model.flat()
        grad = np.zeros_like(base)

        for i in range(base.size):
            v = base.copy()
            v[i] += self.PS
            self.model.set_flat(v)
            lp = infidelity(self.model.unitary(), target)

            v[i] -= 2 * self.PS
            self.model.set_flat(v)
            lm = infidelity(self.model.unitary(), target)

            grad[i] = 0.5 * (lp - lm)

        for r in self.regularizers:
            grad += r.grad(base)

        step = self.opt.step(grad)
        self.model.set_flat(base - step)

        return infidelity(self.model.unitary(), target)

# ============================================================
# Qiskit / UnitarySynthesisPlugin
# ============================================================

class PSFUnitarySynthesisPlugin(UnitarySynthesisPlugin):
    """
    Research-oriented numerical unitary synthesis plugin
    emphasizing hardware-agnostic structure and smooth controls.
    """

    @property
    def max_qubits(self) -> int: return 2
    @property
    def min_qubits(self) -> int: return 2
    @property
    def supported_bases(self) -> list[list[str]]:
        return [["rx", "ry", "rz", "rzz"]]

    def run(self, unitary: np.ndarray, **options) -> QuantumCircuit:
        depth = options.get("depth", 3)
        iters = options.get("iters", 120)

        model = TwoQubitRZZModel(depth=depth, seed=0)
        opt = AdamOptimizer(model.flat().size, AdamConfig(lr=0.2))

        regs = [
            BoundedProjectiveRegularizer(),
            SparsitySmoothnessRegularizer(
                beta_l1=5e-3,
                beta_tv=5e-3,
                shape_angles=model.angles.shape,
                shape_taus=model.taus.shape
            ),
        ]

        trainer = NumericalSynthesizer(model, opt, regs)

        for _ in range(iters):
            trainer.step(unitary)

        qc = QuantumCircuit(2)
        a0 = model.angles[0]
        for q in range(2):
            qc.append(RXGate(a0[q, 0]), [q])
            qc.append(RYGate(a0[q, 1]), [q])
            qc.append(RZGate(a0[q, 2]), [q])

        for k in range(model.depth):
            qc.append(RZZGate(model.taus[k]), [0, 1])
            a = model.angles[k + 1]
            for q in range(2):
                qc.append(RXGate(a[q, 0]), [q])
                qc.append(RYGate(a[q, 1]), [q])
                qc.append(RZGate(a[q, 2]), [q])

        return qc
