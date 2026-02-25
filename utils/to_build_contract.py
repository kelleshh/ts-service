from iep_contract_builder import build_fit_request_payload

contract = build_fit_request_payload(
    iep_rows = [], # пока копипастом для иллюстрации
    # СЮДА ВСТАВИТЬ НАДО
    endpoint = 'fit_auto',
    horizon = 1,
    date_key = 'date',
    drop_keys = ('dataset',),
    timestamp_col = 'ds',
    target_col = 'y',
    target_source_col = "Международные резервы Российской Федерации (еженедельные данные)",
#    exogenous_source_cols = [
#        "Внешняя торговля Российской Федерации услугами по основным странам-партнерам: Импорт | Австрия",
#        "Внешняя торговля Российской Федерации услугами по основным странам-партнерам: Импорт | Азербайджан"
#    ],
    ts =
    {
        'horizon': 1,
        'features': {
            'lags': [1, 2, 3],
            'rolling_mean_windows': [3],
            'rolling_std_windows': [3],
            'rolling_min_windows': [2],
            'rolling_max_windows': [2],
            'diff_lags': [1, 2],
        },
        'split': {
            'valid_fraction': 0.2,
            'min_valid_size': 8,
        },
    },
    training = {
        'model_params': {
            #'n_estimators': 200,
            'random_state': 42,
            'n_jobs': 1,
        },
        'metrics': ['rmse', 'mae', 'mape'],
        'primary_metric': 'rmse',
    },
    tuning = {
        'n_trials': 150,
        'timeout_sec': 45,
    }
)

import json

with open('utils/created_contracts/contract.txt', 'x', encoding='utf-8') as f:
    json.dump(contract, f, ensure_ascii=False, indent=2)