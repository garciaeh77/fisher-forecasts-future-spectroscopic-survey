import numpy as np

from mini_p1d.pipeline import PipelineConfig, run_pipeline
from mini_p1d.power import TabulatedMatterPower, demo_cdm_power


def test_end_to_end_pipeline():
    k = np.geomspace(1.0e-3, 50.0, 3000)
    model = TabulatedMatterPower(k, demo_cdm_power(k))
    config = PipelineConfig(
        length_mpch=100.0,
        n_pixels=128,
        n_skewers=64,
        target_mean_flux=0.80,
        seed=4,
    )
    result = run_pipeline(model, config)
    assert abs(result["mean_flux"] - 0.80) < 1.0e-10
    assert np.all(np.isfinite(result["p1d_flux_mpc_h"]))
    assert result["fgpa_A"] > 0

