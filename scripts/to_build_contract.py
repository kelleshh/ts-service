from utils.iep_contract_builder import build_fit_request_payload

import json
import os
import math

'''
ЭТО СЛУЖЕБНЫЙ КОНСТРУКТОР ДЛЯ СОЗДАНИЯ JSON-КОНТРАКТА
ИЗ СЫРОГО ВРЕМЕННОГО РЯДА (JSON-ФАЙЛ)
'''

RAW_FILE = 'scripts/raw_timeseries/ru_reserves.json'   # <- здесь менять файл сырой


base = os.path.splitext(os.path.basename(RAW_FILE))[0]
OUT_FILE = f'scripts/created_contracts/{base}_contract.json'

with open(RAW_FILE, 'r', encoding='utf-8') as f:
    iep_rows = json.load(f)

if isinstance(iep_rows, dict) and 'rows' in iep_rows:
    iep_rows = iep_rows['rows']


iep_rows = [
    row for row in iep_rows
]

contract = build_fit_request_payload(
    iep_rows=iep_rows,  # type: ignore
    endpoint='fit_auto',
    horizon=1,
    date_key='date',
    drop_keys=('dataset',),
    timestamp_col='ds',
    target_col='y',
    target_source_col= "Международные резервы Российской Федерации (еженедельные данные)",
    training={
        'primary_metric': 'rmse',
        'metrics': ['rmse', 'mae', 'mape'],
        'model_params': {
            'n_estimators': 300,
            'learning_rate': 0.04,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
        },
        'early_stopping_rounds': 50,
        'n_estimators_cap': 2000,
    },
    idempotency_key=f'{base}:fit:h1',
    tuning={
        'n_trials': 100,
        'timeout_sec': 60,
    }
)

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(contract, f, ensure_ascii=False, indent=2)

print('OK:', OUT_FILE)