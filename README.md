# Reinforcement Learning for PQC Noise Mitigation

This repository contains a simple example of using reinforcement learning to tune a parameterized quantum circuit (PQC) in the presence of noise. The `pqc_rl.py` script uses Qiskit to simulate a single-qubit circuit with depolarizing noise. A lightweight policy gradient updates the rotation angles to maximize the probability of measuring `|0⟩` despite noise.

## Requirements
- Python 3
- `qiskit` installed in your environment
- `numpy`

## Usage
Run training for the PQC agent:

```bash
python pqc_rl.py
```

The script prints the final learned parameters and the average reward from the last few episodes.
