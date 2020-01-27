# We will simulate the 18 dynamical regimes that are defined in this list, for various topologies.

dynamical_regimes = list()

for cp in [1, 3]:
    for pk in [1./6, 1./18]: # 1./2,
        for dkh in [1./100, 1./10, 1]:
            dynamical_regimes.append({"consumer/producer": cp, "P/K": pk, "D^2/KH": dkh})
