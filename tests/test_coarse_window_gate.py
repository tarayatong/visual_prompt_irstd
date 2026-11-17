import numpy as np
from visual_prompt.core.coarse_window_gate import (
    WINDOWS, coverage, window_maxima, instance_survival_scores, calibrated_threshold)


def test_gate_boundaries_and_empty_selection():
    assert len(WINDOWS) == 49
    assert not coverage([]).any()
    assert coverage(range(49)).all()
    score = np.zeros((512, 512), np.float32)
    score[511, 511] = 2
    maxima = window_maxima(score)
    assert np.flatnonzero(maxima >= 1).tolist() == [48]
    assert coverage([48])[511, 511]


def test_calibration_matches_actual_instance_coverage():
    rng = np.random.default_rng(7)
    maxima = rng.random(49)
    labels = np.zeros((512, 512), np.int32)
    labels[0, 0] = 1
    labels[250:254, 250:254] = 2
    labels[511, 511] = 3
    scores = instance_survival_scores(labels, maxima)
    for threshold in list(maxima) + [2.]:
        retained = coverage(np.flatnonzero(maxima >= threshold))
        actual = [bool(retained[labels == iid].any()) for iid in range(1, 4)]
        assert actual == [value >= threshold for value in scores]
    assert calibrated_threshold(scores, 1.) == min(scores)
    assert calibrated_threshold(scores, .5) == sorted(scores, reverse=True)[1]
