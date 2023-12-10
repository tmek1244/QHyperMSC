import yaml
import copy
import numpy as np

from QHyper.solvers import solver_from_config
from QHyper.util import weighted_avg_evaluation


def run_experiment(config, samples_per_epoch, elite_frac):
    _config = copy.deepcopy(config)
    _config['solver']['hyper_optimizer'][
        'samples_per_epoch'] = samples_per_epoch
    _config['solver']['hyper_optimizer'][
        'elite_frac'] = elite_frac
    _config['solver']['hyper_optimizer'][
        'epochs'] = 10_000 // samples_per_epoch
    solver = solver_from_config(_config)
    results = solver.solve()

    return weighted_avg_evaluation(
        results.results_probabilities, solver.problem.get_score)


if __name__ == '__main__':
    file_name = "cem_test.yaml"

    with open(f"configs/{file_name}", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    samples_per_epochs = [100, 500, 1000, 2000]
    elite_fracs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    for samples_per_epoch in samples_per_epochs:
        for elite_frac in elite_fracs:
            print(f"Running experiment with samples_per_epoch="
                  f"{samples_per_epoch}, elite_frac={elite_frac}")

            results = [
                run_experiment(samples_per_epoch, elite_frac)
                for _ in range(10)
            ]
            print(f"Results: {np.mean(results)} +- {np.std(results)}")

            with open(f"results/{file_name}", 'a') as f:
                f.write(f"{samples_per_epoch},{elite_frac},"
                        f"{np.mean(results)},{np.std(results)}\n")
