import pytest

pytest.importorskip('polars')

from fastapi.testclient import TestClient

from tsfit.main import create_app


def test_fit_endpoint_returns_metrics() -> None:
    app = create_app()
    client = TestClient(app)

    rows = []
    for i in range(30):
        rows.append({'sid': 'A', 'ts': f'2024-01-{i+1:02d}', 'y': float(i + 1)})

    payload = {
        'dataset': rows,
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
            'xgb_params': {'n_estimators': 10, 'max_depth': 3},
            'metrics': ['rmse', 'mae'],
            'primary_metric': 'rmse',
            'tuning': {'enabled': False, 'n_trials': 5, 'timeout_sec': 10, 'early_stopping_rounds': 10},
        },
    }

    r = client.post('/fit', json=payload)
    assert r.status_code == 201
    body = r.json()
    assert 'run_id' in body
    assert body['metrics'] is not None
    assert 'rmse' in body['metrics']
    assert 'mae' in body['metrics']
