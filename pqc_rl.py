import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error


def create_pqc(theta1, theta2):
    circuit = QuantumCircuit(1)
    circuit.ry(theta1, 0)
    circuit.rz(theta2, 0)
    return circuit


def simulate(circuit, noise_model=None):
    backend = Aer.get_backend('aer_simulator')
    if noise_model:
        backend.set_options(noise_model=noise_model)
    job = execute(circuit, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()
    prob_0 = counts.get('0', 0) / 1024
    return prob_0


def target_fidelity(prob_0):
    return prob_0  # target state is |0>


class RLPQC:
    def __init__(self, lr=0.1, sigma=0.5):
        self.mu = np.zeros(2)
        self.sigma = sigma
        self.lr = lr
        self.noise_model = NoiseModel()
        error = depolarizing_error(0.05, 1)
        self.noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz'])

    def sample_action(self):
        return np.random.normal(self.mu, self.sigma)

    def run_episode(self):
        theta = self.sample_action()
        circuit = create_pqc(theta[0], theta[1])
        prob_0 = simulate(circuit, noise_model=self.noise_model)
        reward = target_fidelity(prob_0)
        self.mu += self.lr * reward * (theta - self.mu)
        return reward


def train(episodes=100):
    agent = RLPQC()
    rewards = []
    for _ in range(episodes):
        r = agent.run_episode()
        rewards.append(r)
    print('Final parameters:', agent.mu)
    print('Average reward:', np.mean(rewards[-10:]))


if __name__ == '__main__':
    train()
