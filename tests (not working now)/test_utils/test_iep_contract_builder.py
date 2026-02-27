import pytest

from scripts.utils.iep_contract_builder import build_fit_request_payload, ContractBuildError


def test_build_fit_payload_single_value_column() -> None:
    rows = [
        {
            'dataset': '379709',
            'date': '01.01.2019',
            'Показатель': '16193523.519',
        },
        {
            'dataset': '379709',
            'date': '01.02.2019',
            'Показатель': '18787196.084',
        },
    ] # TODO: вынести в фикстуру по хорошему но пофиг пока

    payload = build_fit_request_payload(rows, endpoint='fit')

    assert set(payload.keys()) == {'dataset', 'dataset_schema', 'ts', 'training', 'idempotency_key'}
    assert len(payload['dataset']) == 2

    r0 = payload['dataset'][0]
    assert 'dataset' not in r0
    assert r0['ds'] == '2019-01-01T00:00:00Z'
    assert isinstance(r0['y'], float)
    assert r0['y'] == pytest.approx(16193523.519)

    schema = payload['dataset_schema']
    assert schema['timestamp_col'] == 'ds'
    assert schema['target_col'] == 'y'
    assert schema['series_id_col'] is None
    assert schema['exogenous_cols'] == []


def test_build_fit_auto_payload_adds_tuning() -> None:
    rows = [
        {'dataset': '1', 'date': '2019-01-01', 'val': 1},
        {'dataset': '1', 'date': '2019-01-02', 'val': 2},
    ]

    payload = build_fit_request_payload(rows, endpoint='fit_auto', target_source_col='val')

    assert 'tuning' in payload
    assert payload['tuning']['n_trials'] == 50


def test_multiple_value_columns_require_target_source_col() -> None:
    rows = [
        {'dataset': '1', 'date': '01.01.2019', 'a': 1, 'b': 2},
        {'dataset': '1', 'date': '01.02.2019', 'a': 3, 'b': 4},
    ]

    with pytest.raises(ContractBuildError):
        build_fit_request_payload(rows)


def test_invalid_date_raises_contract_build_error() -> None:
    rows = [
        {'dataset': '1', 'date': '01-01-2019', 'val': 1},
    ]

    with pytest.raises(ContractBuildError):
        build_fit_request_payload(rows, target_source_col='val')
