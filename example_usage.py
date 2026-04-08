import numpy as np
from qiskit.quantum_info import random_unitary
from psf_unitary_synthesis import PSFUnitarySynthesisPlugin

def main():
    print("--- PSF Unitary Synthesis Plugin Example ---")
    
    # 1. Generate a random 2-qubit target unitary (SU(4))
    np.random.seed(42)
    target_unitary = random_unitary(4).data
    
    print("Target Unitary generated.")
    print("Synthesizing with Soft Physical Priors (L1 + TV)...")
    
    # 2. Initialize the plugin and run synthesis
    plugin = PSFUnitarySynthesisPlugin()
    
    # Run with custom options (depth=3, iters=150)
    qc = plugin.run(target_unitary, depth=3, iters=150)
    
    # 3. Output the result
    print("\nSynthesis Complete! Resulting Circuit:")
    print(qc.draw())
    
    print("\nGate counts:")
    print(qc.count_ops())

if __name__ == "__main__":
    main()
