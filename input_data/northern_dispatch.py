import numpy as np
import pandas

from input_data import dynamical_regimes
from src.parameters_and_fixed_points import improve_sync, northern_admittance, sync_condition

number_of_nodes = 48
nob = 100
dr = dynamical_regimes[1]
alphas = np.linspace(0, 0.3, 31)

northernY = 1.j * northern_admittance("northern.json")
number_of_nodes = northernY.shape[0]

optimal_P = dict.fromkeys(np.unique([dr['consumer/producer'] for dr in dynamical_regimes]), 0)

for dr in dynamical_regimes:
    cp = dr["consumer/producer"]
    pk = dr["P/K"]
    dkh = dr["D^2/KH"]

    Y = 1. * northernY / (pk * np.mean(np.abs(northernY)[northernY.nonzero()]))

    if optimal_P[cp] is 0:
        assert number_of_nodes % (cp + 1) == 0
        n_gen = int(number_of_nodes / (cp + 1))
        Pg = (cp + 1.) / 2.
        Pc = - 1. * Pg / cp
        P_init = np.ones(number_of_nodes) * Pc
        idx = np.random.choice(range(number_of_nodes), n_gen, replace=False)
        P_init[idx] = Pg

        P = improve_sync(Y, P_init, max_trials=1000, flow_limit=1.)
        optimal_P[cp] = np.copy(P)
    else:
        P = optimal_P[cp]

    print(pk, sync_condition(Y, P))

df = pandas.DataFrame.from_dict(optimal_P)

print(df.head())

df.to_csv("northern_dispatch.csv")