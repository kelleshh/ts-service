from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from tsfit.domain.value_objects import (
    DatasetSchemaValueObject,
    TimeSeriesConfigValueObject,
    TrainingConfigValueObject,
    TuningConfigValueObject,
)


'''
Хэширование полезной нагрузки (payload) для идемпотентности
'''


def canonical_rows(
    rows: list[dict[str, Any]],
    schema: DatasetSchemaValueObject,
) -> list[dict[str, Any]]:
    '''Приводит строки датасета к каноничному порядку.
    Хэширование зависит от порядка строк. Чтобы хэш не менялся из-за случайного
    перемешивания JSON, мы сортируем строки:
    - если есть series_id_col: (series_id, timestamp)
    - иначе: (timestamp)
    Значения приводим к строке для стабильности сортировки.
    '''

    ts = schema.timestamp_col
    sid = schema.series_id_col

    if sid:
        return sorted(rows, key=lambda r: (str(r.get(sid)), str(r.get(ts))))
    return sorted(rows, key=lambda r: str(r.get(ts)))


def _hash_payload(payload: dict[str, Any]) -> str:
    '''
    Считает SHA-256 от JSON-представления payload
    '''

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def hash_fit_payload(
    *,
    dataset_rows: list[dict[str, Any]],
    dataset_schema: DatasetSchemaValueObject,
    time_series: TimeSeriesConfigValueObject,
    training: TrainingConfigValueObject,
) -> str:
    '''
    Хэш payload для сценария /fit (без автоподбора)
    '''

    rows = canonical_rows(dataset_rows, dataset_schema)
    payload = {
        'mode': 'fit',
        'dataset_rows': rows,
        'dataset_schema': asdict(dataset_schema),
        'time_series': asdict(time_series),
        'training': asdict(training),
        'payload_version': 1,
    }
    return _hash_payload(payload)


def hash_fit_auto_payload(
    *,
    dataset_rows: list[dict[str, Any]],
    dataset_schema: DatasetSchemaValueObject,
    time_series: TimeSeriesConfigValueObject,
    training: TrainingConfigValueObject,
    tuning: TuningConfigValueObject,
) -> str:
    '''
    Хэш payload для сценария /fit_auto (с автоподбором)
    '''

    rows = canonical_rows(dataset_rows, dataset_schema)
    payload = {
        'mode': 'auto',
        'dataset_rows': rows,
        'dataset_schema': asdict(dataset_schema),
        'time_series': asdict(time_series),
        'training': asdict(training),
        'tuning': asdict(tuning),
        'payload_version': 1,
    }
    return _hash_payload(payload)
