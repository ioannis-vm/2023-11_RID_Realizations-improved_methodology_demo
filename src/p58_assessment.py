"""
Loss estimation with FEMA P-58, using pelicun
"""

import concurrent.futures
from itertools import product
import tqdm
import pickle
import numpy as np
import pandas as pd
from pelicun.assessment import Assessment
from src.models import Model_1_Weibull
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.util import store_info


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


sa_t1 = {
    '1': 0.2066104302916705,
    '2': 0.5568180337401586,
    '3': 0.8898019882446327,
    '4': 1.2147414530074077,
    '5': 1.588363803080204,
    '6': 1.9547124114163368,
    '7': 2.324632990424402,
    '8': 2.711464597513382,
}

system = 'scbf'
stories = '9'
rc = 'ii'
edp_dataset = remove_collapse(
    load_dataset('data/edp.parquet')[0], drift_threshold=0.06
)[(system, stories, rc)]
num_hz = len(edp_dataset.index.get_level_values('hz').unique())


def run_case(hz, rid_method):
    # asmt = Assessment({"PrintLog": False, "Seed": 1})
    asmt = Assessment({"PrintLog": False})  # no seed, debug
    asmt.stories = 9

    # ---------------------------------- #
    # Building Response Realizations     #
    # ---------------------------------- #

    def edps():
        def process_demands(hz):
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

        all_demands = {}
        all_rid_demands = {}
        for i_hz in [f'{i+1}' for i in range(num_hz)]:
            demands, rid_demands = process_demands(i_hz)
            all_demands[i_hz] = demands
            all_rid_demands[i_hz] = rid_demands
        demands = all_demands[hz]
        rid_demands = all_rid_demands[hz]
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
            demand_sample_ext = pd.concat([demand_sample, rid], axis=1)
        elif rid_method == 'FEMA P-58 optimized':
            pid = demand_sample["PID"]
            rid = asmt.demand.estimate_RID(pid, {"yield_drift": 0.01082})
            demand_sample_ext = pd.concat([demand_sample, rid], axis=1)
        elif rid_method == 'Conditional Weibull':
            # fit models
            rid_demands_df = pd.concat(
                all_rid_demands.values(),
                keys=all_rid_demands.keys(),
                axis=1,
                names=('hz', 'loc', 'dir'),
            )
            pid_demands_df = pd.concat(
                [all_demands[f'{i+1}']['PID'] for i in range(num_hz)],
                keys=all_demands.keys(),
                axis=1,
                names=('hz', 'loc', 'dir'),
            )
            models = {}
            for i_loc in [f'{i+1}' for i in range(9)]:
                analysis_pids = (
                    pid_demands_df.xs(i_loc, level='loc', axis=1)
                    .drop('Units')
                    .astype(float)
                    .stack()
                    .stack()
                )
                analysis_rids = (
                    rid_demands_df.xs(i_loc, level='loc', axis=1).stack().stack()
                )
                # import matplotlib.pyplot as plt
                # plt.scatter(analysis_rids, analysis_pids)
                # plt.show()
                assert np.all(analysis_pids.index == analysis_rids.index)
                model = Model_1_Weibull()
                model.add_data(analysis_pids.values, analysis_rids.values)
                model.fit(method='quantiles')
                # fig, ax = plt.subplots()
                # model.plot_model(ax=ax)
                # plt.show()
                models[i_loc] = model
            # simulate data
            pid_df = demand_sample['PID']
            rid_cols = []
            num_points = len(demand_sample['PID'].iloc[:, 0])
            uniform_sample = np.random.uniform(0.00, 1.00, num_points)
            for col in pid_df.columns:
                # generate samples
                pid_sample = demand_sample['PID', *col].values
                assert len(pid_sample) == num_points
                model = models[col[0]]
                model.uniform_sample = uniform_sample
                rid_sample = model.generate_rid_samples(pid_sample)
                rid_cols.append(rid_sample)
            rid = pd.DataFrame(rid_cols, index=pid_df.columns).T
            rid = pd.concat([rid], axis=1, keys=['RID'])
            demand_sample_ext = pd.concat([demand_sample, rid], axis=1)
        elif rid_method == 'Empirical':
            raw_demands_df = pd.concat(
                (
                    demands.drop('Units').astype(float),
                    pd.concat((rid_demands,), keys=('RID',), axis=1),
                ),
                axis=1,
            )
            vals = raw_demands_df.values
            num_rows, num_columns = np.shape(vals)
            random_idx = np.random.choice(num_rows, num_realizations, replace=True)
            new_vals = vals[random_idx, :]
            demand_sample_ext = pd.DataFrame(
                new_vals, columns=raw_demands_df.columns
            )
        else:
            raise ValueError(f'Invalid rid_method: {rid_method}')
        sa_t = sa_t1[hz]
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
    methods = [
        'FEMA P-58',
        'Conditional Weibull',
        'Empirical',
    ]
    args_list = list(product(hzs, methods))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_case, *args): args for args in args_list}
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(args_list)
        ):
            results.append(future.result())

    hz_list = [int(res[0]) for res in results]
    method_list = [res[1] for res in results]
    agg_df_list = [res[2] for res in results]
    demands_list = [res[3] for res in results]
    demand_sample_ext_list = [res[4] for res in results]
    df = pd.concat(agg_df_list, axis=1, keys=zip(method_list, hz_list))
    df = df.sort_index(axis=1)

    with open(
        store_info('extra/improved_methodology_demo/results/out.pcl'), 'wb'
    ) as f:
        pickle.dump(df, f)
