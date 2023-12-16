import sys
import numpy as np
import multiprocessing as mp

from QHyper.solvers import solver_from_config
from QHyper.util import weighted_avg_evaluation


problems = {
    'kp_1': {
        'type': 'knapsack',
        'max_weight': 3,
        'items': [[1, 1], [2, 1], [3, 1], [4, 1]]
    },
    'kp_2': {
        'type': 'knapsack',
        'max_weight': 2,
        'items': [[1, 2], [1, 2], [1, 1]]
    },
    'kp_3': {
        'type': 'knapsack',
        'max_weight': 4,
        'items': [[1, 2], [3, 2], [2, 3], [1, 1]]
    },
    'kp_4': {
        'type': 'knapsack',
        'max_weight': 3,
        'items': [[3, 5], [1, 2], [1, 1]]
    },
    'kp_5': {
        'type': 'knapsack',
        'max_weight': 1,
        'items': [[1, 2], [1, 1]]
    },
    'tsp_1': {
        'type': 'tsp',
        'number_of_cities': 3,
        'cities_coords': [[0, 0], [0, 1], [1, 0]]
    },
    'tsp_2': {
        'type': 'tsp',
        'number_of_cities': 2,
        'cities_coords': [[0, 0], [1, 1]]
    },
    'tsp_3': {
        'type': 'tsp',
        'number_of_cities': 2,
        'cities_coords': [[0, 0], [0.4, 0.6]]
    },
    'tsp_4': {
        'type': 'tsp',
        'number_of_cities': 3,
        'cities_coords': [[0, 0], [1, 1], [0.5, 0.7]]
    },
    'tsp_5': {
        'type': 'tsp',
        'number_of_cities': 3,
        'cities_coords': [[0.2, 0.7], [0.4, 0.8], [0.8, 0.1]]
    },
}


solvers = {
    'wfqaoa': {
        "type": "vqa",
        "optimizer": {
            "type": "scipy",
            "maxfun": 200,
        },
        "pqc": {
            "type": "wfqaoa",
            "layers": 5,
            "penalty": 3,
        },
    },
    'qaoa': {
        "type": "vqa",
        "optimizer": {
            "type": "scipy",
            "maxfun": 200,
        },
        "pqc": {
            "type": "qaoa",
            "layers": 5,
        },
    },
}


if __name__ == "__main__":
    PROBLEM = sys.argv[1]
    SOLVER = sys.argv[2]

    np.random.seed(0)

    angles = np.random.rand(1_000, 5, 2)

    def run_solver(angle):
        if 'kp' in PROBLEM:
            hyper_args = [1, 2.5, 2.5]
            penalty = 0
        elif 'tsp' in PROBLEM:
            cities = problems[PROBLEM]['number_of_cities']
            hyper_args = [1] + (cities + 1) * [2]
            penalty = 5
            solvers['wfqaoa']['pqc']['penalty'] = penalty

        solver = solver_from_config({
            'solver': solvers[SOLVER],
            'problem': problems[PROBLEM],
        })
        results = solver.solve({
                'angles': angle,
                'hyper_args': hyper_args
            })
        value = weighted_avg_evaluation(
            results.results_probabilities, solver.problem.get_score,
            limit_results=30, penalty=penalty)
        # print(angle, value)
        with open(f'./results/{PROBLEM}_{SOLVER}.txt', 'a') as f:
            f.write(f'{value}\n')

    with mp.Pool(10) as pool:
        pool.map(run_solver, angles)
