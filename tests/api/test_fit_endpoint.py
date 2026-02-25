import pytest

pytest.importorskip('polars')

from fastapi.testclient import TestClient

from tsfit.main import create_app

def _make_rows(n: int = 30) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({'sid': 'A', 'ts': f'2024-01-{i+1:02d}', 'y': float(i + 1)})
    return rows


def test_fit_endpoint_returns_metrics() -> None:
    app = create_app()
    client = TestClient(app)

    payload = {
        'dataset': _make_rows(30),
        'dataset_schema': {
            'timestamp_col': 'ts',
            'target_col': 'y',
            'series_id_col': 'sid',
            'exogenous_cols': [],
        },
        'ts': {
            'horizon': 1,
            'features': {
                'lags': [1, 2, 3],
                'rolling_mean_windows': [3],
                'rolling_std_windows': [3],
                'rolling_min_windows': [],
                'rolling_max_windows': [],
                'diff_lags': [],
            },
            'split': {'valid_fraction': 0.2, 'min_valid_size': 3},
        },
        'training': {
            'model_params': {'n_estimators': 10, 'max_depth': 3},
            'metrics': ['rmse', 'mae'],
            'primary_metric': 'rmse',
        },
    }

    r = client.post('/fit', json=payload)
    assert r.status_code == 201, r.text
    body = r.json()

    assert 'run_id' in body
    assert body['metrics'] is not None
    assert 'rmse' in body['metrics']
    assert 'mae' in body['metrics']




def test_fit_auto_endpoint_returns_metrics_and_tuning_report() -> None:
    app = create_app()
    client = TestClient(app)

    payload = {
        'dataset': _make_rows(40),
        'dataset_schema': {
            'timestamp_col': 'ts',
            'target_col': 'y',
            'series_id_col': 'sid',
            'exogenous_cols': [],
        },
        'ts': {
            'horizon': 1,
            'features': {
                'lags': [1, 2, 3],
                'rolling_mean_windows': [3],
                'rolling_std_windows': [3],
                'rolling_min_windows': [],
                'rolling_max_windows': [],
                'diff_lags': [],
            },
            'split': {'valid_fraction': 0.2, 'min_valid_size': 3},
        },
        'training': {
            'model_params': {'n_estimators': 30, 'max_depth': 3},
            'metrics': ['rmse', 'mae'],
            'primary_metric': 'rmse',
        },
        'tuning': {
            # для тестов держим маленькие значения чтобы быстро отрабатывало
            'n_trials': 3,
            'timeout_sec': 5,
        },
    }

    r = client.post('/fit/auto', json=payload)
    assert r.status_code == 201, r.text
    body = r.json()

    assert 'run_id' in body
    assert body['metrics'] is not None
    assert 'rmse' in body['metrics']

    assert 'tuning_report' in body
    assert body['tuning_report'] is not None
    assert 'stage2' in body['tuning_report']
    assert body['tuning_report']['stage2'] is not None
    assert 'best_params' in body['tuning_report']['stage2']