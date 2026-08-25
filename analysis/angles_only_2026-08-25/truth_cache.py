"""Load cached truth from out/*.csv instead of re-propagating.

Avoids mc_fgo.propagate_truth's fixed /tmp path, which races when two
processes run concurrently.  Returns the same tuple propagate_truth does.
"""
import os
import numpy as np
import pandas as pd
from fgo_pipeline import load_propagator_output, load_config_parameters
from Orbit_FGO import ric_to_eci
import mc_fgo


def load_truth(config_path, tag):
    combined = os.path.abspath(f"out/mc_fgo_truth_{tag}.csv")
    pre = os.path.abspath(f"out/mc_fgo_truth_pre_{tag}.csv")
    if not (os.path.exists(combined) and os.path.exists(pre)):
        return mc_fgo.propagate_truth(config_path, tag)

    cp, _ = load_config_parameters(config_path)
    delta_v_ric = np.array(cp['delta_v_ric'], dtype=float)
    df_pre = pd.read_csv(pre)
    manoeuvre_state = df_pre[['x', 'y', 'z', 'vx', 'vy', 'vz']].iloc[-1].values
    t_star_true = float(df_pre['tSec'].iloc[-1])
    delta_v_eci = ric_to_eci(delta_v_ric, manoeuvre_state)
    truth_states, times, dt = load_propagator_output(combined)
    return (truth_states, times, dt, delta_v_ric, delta_v_eci,
            manoeuvre_state, t_star_true)
