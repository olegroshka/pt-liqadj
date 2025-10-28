from ptliq.training.gat_loop import sched_alpha, sched_lambda_huber, sched_lambda_noncross, sched_lambda_wmono, sched_beta_q90


def test_schedule_values():
    # Epochs 0..10 (0-based indexing)
    alphas = [sched_alpha(e) for e in range(0, 11)]
    hubers = [sched_lambda_huber(e) for e in range(0, 11)]
    noncross = [sched_lambda_noncross(e) for e in range(0, 11)]
    wmono = [sched_lambda_wmono(e) for e in range(0, 11)]
    betas = [sched_beta_q90(e) for e in range(0, 11)]

    # 0-2
    for e in range(0, 3):
        assert alphas[e] == 0.0
        assert hubers[e] == 1.0
        assert noncross[e] == 0.10
        assert wmono[e] == 0.0
        assert betas[e] == 0.0

    # 3-5: linear ramps
    assert abs(alphas[3] - 0.1) < 1e-9
    assert abs(hubers[3] - 1.0) < 1e-9
    assert 0.1 < alphas[4] < 0.6
    assert 0.3 < hubers[4] < 1.0
    assert abs(alphas[5] - 0.6) < 1e-9
    assert abs(hubers[5] - 0.3) < 1e-9

    # >=6: fixed
    for e in range(6, 11):
        assert abs(alphas[e] - 1.0) < 1e-9
        assert abs(hubers[e] - 0.2) < 1e-9
        assert noncross[e] == 0.10
        assert wmono[e] == 0.10
        assert betas[e] == 0.0
