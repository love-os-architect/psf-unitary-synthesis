# Physically-Regularized Numerical 2Q Unitary Synthesis (Research Plugin)

Hello QDK team,

I've been exploring a research-oriented numerical unitary synthesis approach for two-qubit gates. I wanted to share this concept here, as its design goals seem to align closely with QDK's focus on hardware-agnostic, resource-aware, and fault-tolerant workflows.

**Repository:** [https://github.com/love-os-architect/README]

### Summary
This method synthesizes an arbitrary SU(4) unitary into an alternating structure of local rotations and RZZ entanglers. Unlike standard analytical decompositions, it uses differentiable parameter-shift gradients combined with **physically motivated soft priors**.

### Key Design Goals
* **Hardware-agnostic gate structure:** (RX/RY/RZ + RZZ) with no coupling-map assumptions.
* **Friendly to future FT compilation flows:** Designed to handle "soft" constraints rather than forcing exact, rigid mathematical decompositions.
* **Explicit separation of concerns:** The Model, Optimizer, Regularization policy, and Metric are decoupled for easy research replacement.

### Regularization Policies
This work intentionally goes beyond pure fidelity maximization by introducing soft physical priors to manage hardware friction:
1. **Bounded projective regularization:** Softly constrains parameters without hard clipping, avoiding optimization instability while remaining analytically differentiable.
2. **Sparsity & smoothness priors (L1 + Total Variation):** Encourages low-dissipation and pulse-friendly parameter profiles, acting as a lightweight proxy for hardware control costs.

### Intended Scope & Integration
* This is intended as a **research/prototyping plugin**, not a deterministic replacement for analytic decompositions (like KAK).
* It is designed to peacefully coexist with existing synthesis passes.

### Why share this here?
I believe this approach complements Azure Quantum's resource estimation and future FT planning. It explores synthesis under *soft physical constraints*—embracing the inevitable noise and variations of real hardware—rather than seeking perfect theoretical decomposition. 

I would be very interested to hear your thoughts on this approach. Is there a specific extension point within the current QDK architecture where experimenting with such hardware-informed synthesis heuristics would be most valuable? 

Any feedback, discussion, or redirection would be greatly appreciated!
