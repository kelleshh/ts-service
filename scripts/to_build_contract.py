from utils.iep_contract_builder import build_fit_request_payload

import json
import os
import math

'''
ЭТО СЛУЖЕБНЫЙ КОНСТРУКТОР ДЛЯ СОЗДАНИЯ JSON-КОНТРАКТА
ИЗ СЫРОГО ВРЕМЕННОГО РЯДА (JSON-ФАЙЛ)
'''

RAW_FILE = 'scripts/raw_timeseries/bonds_yield.json'   # <- здесь менять файл сырой


base = os.path.splitext(os.path.basename(RAW_FILE))[0]
OUT_FILE = f'scripts/created_contracts/{base}_contract.json'

with open(RAW_FILE, 'r', encoding='utf-8') as f:
    iep_rows = json.load(f)

if isinstance(iep_rows, dict) and 'rows' in iep_rows:
    iep_rows = iep_rows['rows']


# ПРЕДПОЛАГАЕТСЯ ЧТО ПРИХОДЯЩИЙ JSON БЕЗ ПРОПУСКОВ. ОБРАБОТКУ ПРОПУСКОВ, ВЫБРОСОВ И Т.Д. ЛУЧШЕ ДЕЛАТЬ В ОТДЕЛЬНОМ ЭНДПОИНТЕ /preprocess, ЧТОБЫ НЕ СМЕШИВАТЬ ОТВЕТСТВЕННОСТИ
iep_rows = [
    row for row in iep_rows
    if all(
        not (
            (isinstance(v, (int, float)) and not math.isfinite(v)) or
            (isinstance(v, str) and v.strip().lower() in {'nan', 'inf', '-inf'})
        )
        for v in row.values()
    )
]

contract = build_fit_request_payload(
    iep_rows=iep_rows, # type: ignore
    endpoint='fit',
    horizon=1,
    date_key='date',
    drop_keys=('dataset',),
    timestamp_col='ds',
    target_col='y',
    target_source_col='Долгосрочная доходность по облигациям.',
    ts={
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
    training={
        'model_params': {
            'n_estimators': 300,
            'learning_rate': 0.04,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'random_state': 42,
        },
        'metrics': ['rmse', 'mae', 'mape'],
        'primary_metric': 'rmse',
    },
#    tuning={
#        'n_trials': 150,
#        'timeout_sec': 45,
#    },
)

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(contract, f, ensure_ascii=False, indent=2)

print('OK:', OUT_FILE)