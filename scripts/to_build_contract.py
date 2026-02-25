from utils.iep_contract_builder import build_fit_request_payload

import json
import os

'''
ЭТО СЛУЖЕБНЫЙ КОНСТРУКТОР ДЛЯ СОЗДАНИЯ JSON-КОНТРАКТА
ИЗ СЫРОГО ВРЕМЕННОГО РЯДА (JSON-ФАЙЛ)
'''

RAW_FILE = 'scripts/raw_timeseries/micex_rts_index.json'   # <- здесь менять файл сырой

os.makedirs('created_contracts', exist_ok=True)
base = os.path.splitext(os.path.basename(RAW_FILE))[0]
OUT_FILE = f'scripts/created_contracts/{base}_contract.json'

with open(RAW_FILE, 'r', encoding='utf-8') as f:
    iep_rows = json.load(f)
if isinstance(iep_rows, dict) and 'rows' in iep_rows:
    iep_rows = iep_rows['rows']

contract = build_fit_request_payload(
    iep_rows=iep_rows, # type: ignore
    endpoint='fit_auto',
    horizon=1,
    date_key='date',
    drop_keys=('dataset',),
    timestamp_col='ds',
    target_col='y',
    target_source_col='Индекс Public Joint-Stock Company Moscow Exchange MICEX-RTS (MOEX.ME): цена закрытия',
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
            # 'n_estimators': 200,
            'random_state': 42,
            'n_jobs': 1,
        },
        'metrics': ['rmse', 'mae', 'mape'],
        'primary_metric': 'rmse',
    },
    tuning={
        'n_trials': 150,
        'timeout_sec': 45,
    },
)

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(contract, f, ensure_ascii=False, indent=2)

print('OK:', OUT_FILE)