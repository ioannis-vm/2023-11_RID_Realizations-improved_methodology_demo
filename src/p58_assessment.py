"""
Loss estimation with FEMA P-58, using pelicun
"""

import concurrent.futures
from itertools import product
import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pelicun.assessment import Assessment
from src.models import Model_1_Weibull
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from scipy.interpolate import interp1d
from extra.structural_analysis.src.util import read_study_param


def rm_unnamed(string):
    """
    Fix column names after import
    """
    if "Unnamed: " in string:
        return ""
    return string


# ---------------------------------- #
# Initialization                     #
# ---------------------------------- #

num_realizations = 100
yield_drift = 0.40 / 100.00
archetype = 'scbf_9_ii'
collapse_params = (3.589, 0.556)  # lognormal: median (g), beta


system = 'scbf'
stories = '9'
rc = 'ii'
edp_dataset = remove_collapse(load_dataset()[0], drift_threshold=0.06)[
    (system, stories, rc)
]
num_hz = len(edp_dataset.index.get_level_values('hz').unique())

rid_method = 'FEMA P-58'
# rid_method = 'Weibull'


def run_case(hz, rid_method):
    # asmt = Assessment({"PrintLog": False, "Seed": 1})
    asmt = Assessment({"PrintLog": False})  # no seed, debug
    asmt.stories = 9

    # ---------------------------------- #
    # Building Response Realizations     #
    # ---------------------------------- #

    def edps():
        def process_demands():
            demands = (
                edp_dataset[hz].unstack(level=0).unstack(level=0).unstack(level=0)
            ).dropna('columns')
            demands.columns.names = ["type", "loc", "dir"]
            rid_demands = demands['RID']
            demands.drop(["PVb", 'RID'], axis=1, inplace=True)
            # add units
            units = []
            for col in demands.columns:
                if col[0] == "PFA":
                    units.append(["g"])
                elif col[0] == "PFV":
                    units.append(["inps2"])
                elif col[0] == "PID":
                    units.append(["rad"])
                else:
                    raise ValueError(f"Invalid EDP type: {col[0]}")
            units_df = pd.DataFrame(dict(zip(demands.columns, units)))
            units_df.index = ["Units"]
            demands = pd.concat((units_df, demands))
            return demands, rid_demands

        demands, rid_demands = process_demands()
        asmt.demand.load_sample(demands)
        asmt.demand.calibrate_model(
            {
                "ALL": {
                    "DistributionFamily": "lognormal",
                    "AddUncertainty": 0.00,
                },
                "PID": {
                    "DistributionFamily": "lognormal",
                    "TruncateLower": "",
                    "TruncateUpper": "0.06",
                    "AddUncertainty": 0.00,
                },
            }
        )
        asmt.demand.generate_sample({"SampleSize": num_realizations})
        demand_sample = asmt.demand.save_sample()
        if rid_method == 'FEMA P-58':
            pid = demand_sample["PID"]
            rid = asmt.demand.estimate_RID(pid, {"yield_drift": yield_drift})
        elif rid_method == 'Weibull':
            pid_df = demand_sample['PID']
            rid_cols = []
            for col in pid_df.columns:
                # gather inputs
                rid_vals = rid_demands[col].values
                pid_vals = demands['PID', *col].drop('Units').astype(float).values
                # fit a model
                model = Model_1_Weibull()
                model.add_data(pid_vals, rid_vals)
                model.fit(method='mle')
                # generate samples
                pid_sample = demand_sample['PID', *col].values
                rid_sample = model.generate_rid_samples(pid_sample)
                rid_cols.append(rid_sample)
            rid = pd.DataFrame(rid_cols, index=pid_df.columns).T
            rid = pd.concat([rid], axis=1, keys=['RID'])
        else:
            raise ValueError(f'Invalid rid_method: {rid_method}')

        demand_sample_ext = pd.concat([demand_sample, rid], axis=1)

        spectrum = pd.read_csv(
            f"extra/structural_analysis/results/site_hazard/UHS_{hz}.csv",
            index_col=0,
            header=0,
        )
        ifun = interp1d(spectrum.index.to_numpy(), spectrum.to_numpy().reshape(-1))
        base_period = float(
            read_study_param(
                f"extra/structural_analysis/data/{archetype}/period_closest"
            )
        )
        sa_t = float(ifun(base_period))
        demand_sample_ext[("SA", "0", "1")] = sa_t
        # add units to the data
        demand_sample_ext.T.insert(0, "Units", "")
        demand_sample_ext.loc["Units", ["PFA", "SA"]] = "g"
        demand_sample_ext.loc["Units", ["PID", "RID"]] = "rad"
        demand_sample_ext.loc["Units", ["PFV"]] = "inps2"
        demand_sample_ext.loc["Units", ["SA"]] = "g"
        # load back the demand sample
        asmt.demand.load_sample(demand_sample_ext)

        return demands, demand_sample_ext

    demands, demand_sample_ext = edps()

    # ---------------------------------- #
    # Damage Estimation                  #
    # ---------------------------------- #

    def damage():
        cmp_marginals = pd.read_csv(
            "extra/improved_methodology_demo/data/input_cmp_quant.csv", index_col=0
        )
        cmp_marginals_building = pd.DataFrame(
            {
                "Units": ["ea", "ea", "ea"],
                "Location": ["all", "0", "0"],
                "Direction": ["1,2", "1", "1"],
                "Theta_0": ["1", "1", "1"],
            },
            index=["excessiveRID", "collapse", "irreparable"],
        )
        cmp_marginals = pd.concat((cmp_marginals, cmp_marginals_building))

        asmt.asset.load_cmp_model({"marginals": cmp_marginals})
        asmt.asset.generate_cmp_sample()

        damage_db = pd.read_csv(
            "extra/improved_methodology_demo/data/input_damage.csv",
            header=[0, 1],
            index_col=0,
        )

        damage_db.rename(columns=rm_unnamed, level=1, inplace=True)
        damage_db.loc[
            "collapse", [("LS1", "Family"), ("LS1", "Theta_0"), ("LS1", "Theta_1")]
        ] = ["lognormal", collapse_params[0], collapse_params[1]]

        asmt.damage.load_damage_model([damage_db])
        dmg_process = {
            "1_collapse": {"DS1": "ALL_NA"},
            "2_excessiveRID": {"DS1": "irreparable_DS1"},
            "3_D.20.21.013a": {"DS2": "C.30.21.001k_DS1"},
            "4_D.20.31.013b": {"DS2": "C.30.21.001k_DS1"},
            "5_D.40.11.024a": {"DS2": "C.30.21.001k_DS1"},
        }

        asmt.damage.calculate(dmg_process=dmg_process)

        return cmp_marginals

    cmp_marginals = damage()

    # ---------------------------------- #
    # Loss Estimation                    #
    # ---------------------------------- #

    def loss():
        drivers = [f"DMG-{cmp}" for cmp in cmp_marginals.index.unique()]
        drivers = drivers[:-3] + drivers[-2:]
        loss_models = cmp_marginals.index.unique().tolist()[:-3]
        loss_models += [
            "replacement",
        ] * 2
        loss_map = pd.DataFrame(loss_models, columns=["BldgRepair"], index=drivers)

        loss_db = pd.read_csv(
            "extra/improved_methodology_demo/data/input_loss.csv",
            header=[0, 1],
            index_col=[0, 1],
        )
        loss_db.rename(columns=rm_unnamed, level=1, inplace=True)
        loss_db_additional = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                (
                    ("replacement", "Carbon"),
                    ("replacement", "Cost"),
                    ("replacement", "Energy"),
                    ("replacement", "Time"),
                )
            ),
            columns=loss_db.columns,
        )
        loss_db_additional.loc["replacement", ("DV", "Unit")] = [
            "kg",
            "USD_2011",
            "MJ",
            "worker_day",
        ]
        loss_db_additional.loc["replacement", ("Quantity", "Unit")] = ["1 EA"] * 4

        loss_db_additional.loc[("replacement", "Cost"), "DS1"] = [
            "lognormal",
            np.nan,
            33750000.00,
            0.10,
        ]
        loss_db_additional.loc[("replacement", "Time"), "DS1"] = [
            "lognormal",
            np.nan,
            24107.14286,
            0.10,
        ]
        loss_db = loss_db._append(loss_db_additional)

        asmt.bldg_repair.load_model([loss_db], loss_map)

        asmt.bldg_repair.calculate()

        agg_df = asmt.bldg_repair.aggregate_losses()

        return agg_df

    agg_df = loss()

    return hz, rid_method, agg_df, demands, demand_sample_ext


if __name__ == '__main__':
    results = []
    hzs = [f'{i+1}' for i in range(num_hz)]
    methods = ['FEMA P-58', 'Weibull']
    args_list = list(product(hzs, methods))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_case, *args): args for args in args_list}
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(args_list)
        ):
            results.append(future.result())

    hz_list = [res[0] for res in results]
    method_list = [res[1] for res in results]
    agg_df_list = [res[2] for res in results]
    demands_list = [res[3] for res in results]
    demand_sample_ext_list = [res[4] for res in results]

    _, ax = plt.subplots()

    df = pd.concat(agg_df_list, axis=1, keys=zip(method_list, hz_list))
    df = df.sort_index(axis=1)

    df = df['FEMA P-58']

    df_cost = df.xs('repair_cost', axis=1, level=1) / 1e6  # million
    df_cost_describe = df_cost.describe()
    mean = df_cost_describe.loc['mean', :]
    std = df_cost_describe.loc['std', :]
    mean_plus = mean + std
    mean_minus = mean - std

    for i in range(num_hz):
        # sns.ecdfplot(df_cost.iloc[:, i], ax=ax, color='black')
        ax.scatter(
            [i + 1], [df_cost.iloc[:, i].mean()], edgecolors='black', color='none'
        )
        ax.scatter(
            [i + 1],
            [df_cost.iloc[:, i].mean() - df_cost.iloc[:, i].std()],
            edgecolors='black',
            color='none',
        )
        ax.scatter(
            [i + 1],
            [df_cost.iloc[:, i].mean() + df_cost.iloc[:, i].std()],
            edgecolors='black',
            color='none',
        )
        ax.plot(range(1, num_hz + 1), mean.values, color='black', linewidth=0.40)
        ax.plot(
            range(1, num_hz + 1), mean_plus.values, color='black', linewidth=0.40
        )
        ax.plot(
            range(1, num_hz + 1), mean_minus.values, color='black', linewidth=0.40
        )

    df = pd.concat(agg_df_list, axis=1, keys=zip(method_list, hz_list))
    df = df.sort_index(axis=1)

    df = df['Weibull']

    df_cost = df.xs('repair_cost', axis=1, level=1) / 1e6  # million
    df_cost_describe = df_cost.describe()
    mean = df_cost_describe.loc['mean', :]
    std = df_cost_describe.loc['std', :]
    mean_plus = mean + std
    mean_minus = mean - std

    for i in range(num_hz):
        # sns.ecdfplot(df_cost.iloc[:, i], ax=ax, color='red')
        ax.scatter(
            [i + 1], [df_cost.iloc[:, i].mean()], edgecolors='red', color='none'
        )
        ax.scatter(
            [i + 1],
            [df_cost.iloc[:, i].mean() - df_cost.iloc[:, i].std()],
            edgecolors='red',
            color='none',
        )
        ax.scatter(
            [i + 1],
            [df_cost.iloc[:, i].mean() + df_cost.iloc[:, i].std()],
            edgecolors='red',
            color='none',
        )
        ax.plot(range(1, num_hz + 1), mean.values, color='red', linewidth=0.40)
        ax.plot(
            range(1, num_hz + 1), mean_plus.values, color='red', linewidth=0.40
        )
        ax.plot(
            range(1, num_hz + 1), mean_minus.values, color='red', linewidth=0.40
        )

    ax.grid(which='both', linewidth=0.30)
    plt.show()