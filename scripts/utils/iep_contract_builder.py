from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .parsing import ParseError, parse_date_to_iso_z, parse_float_like


@dataclass(frozen=True, slots=True)
class ContractBuildError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _default_ts_config(*, horizon: int) -> dict[str, Any]:
    return {
        'horizon': horizon,
        'features': {
            'lags': [1, 2, 5, 10],
            'rolling_mean_windows': [10],
            'rolling_std_windows': [10],
            'rolling_min_windows': [],
            'rolling_max_windows': [],
            'diff_lags': [1, 2],
        },
        'split': {
            'valid_fraction': 0.2,
            'min_valid_size': 24,
        },
    }


def _default_training_config() -> dict[str, Any]:
    return {
        'model_params': {
            'n_estimators': 400,
            'random_state': 42,
            'n_jobs': 1,
        },
        'metrics': ['rmse', 'mae'],
        'primary_metric': 'rmse',
    }


def _default_tuning_config() -> dict[str, Any]:
    return {
        'n_trials': 50,
        'timeout_sec': 45,
    }


# TODO: добавить нормальную документацию параметров, т.к. это пользовательская утилита
def build_fit_request_payload(
    iep_rows: list[dict[str, Any]],
    *,
    endpoint: Literal['fit', 'fit_auto'] = 'fit',
    horizon: int = 1,
    date_key: str = 'date',
    drop_keys: tuple[str, ...] = ('dataset',),
    timestamp_col: str = 'ds',
    target_col: str = 'y',
    series_id_col: str | None = None,
    target_source_col: str | None = None,
    exogenous_source_cols: list[str] | None = None,
    ts: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    tuning: dict[str, Any] | None = None,
    idempotency_key: str | None = '',
) -> dict[str, Any]:
    '''
    Строит полный JSON-контракт для эндпоинта обучения временных рядов:
    - POST /fit (endpoint='fit')
    - POST /fit/auto (endpoint='fit_auto')

    На вход подаются сырые строки датасета (как на data.iep.ru) например:
    [
      {
        'dataset': '379709',
        'date': '01.01.2019',
        'y': '16193523.519'
      },
      ...
    ]

    Что делает функция:
    1) удаляет служебные ключи (по умолчанию 'dataset') из каждой строки
    2) преобразует дату (обычно 'DD.MM.YYYY') в ISO-строку 'YYYY-MM-DDT00:00:00Z'.
    3) выбирает целевую переменную (target) и при желании внешние факторы (exogenous)
    4) переименовывает колонки так, чтобы соответствовать контракту сервиса:
       timestamp_col='ds', target_col='y'
    5) добавляет разделы dataset_schema, ts, training (а еще tuning для /fit/auto)

    Параметры:
    - target_source_col: имя исходной колонки, которая должна стать игреком.
      если не задано и среди колонок кроме date_key/drop_keys ровно одна - она и будет targetом
    - exogenous_source_cols: список исходных колонок которые пойдут как внешние факторы
      если не задано, то по умолчанию берем все оставшиеся (кроме target)
    '''
    if not iep_rows:
        raise ContractBuildError('iep_rows - пустой список')

    # собираем все ключи, чтобы понять какие колонки есть в данных
    all_keys: set[str] = set()
    for i, row in enumerate(iep_rows):
        if not isinstance(row, dict):
            raise ContractBuildError(f'iep_rows[{i}] должен быть dict, получено: {type(row)}')
        all_keys.update(row.keys())

    if date_key not in all_keys:
        raise ContractBuildError(f'не найдена колонка даты {date_key!r} в входных данных')

    candidate_value_cols = sorted(
        k for k in all_keys
        if k != date_key and k not in drop_keys
    )

    if not candidate_value_cols:
        raise ContractBuildError(
            'не найдено ни одной числовой колонки кроме даты'
            f'Проверь date_key={date_key!r} и drop_keys={drop_keys!r}'
        )

    if target_source_col is None:
        if len(candidate_value_cols) != 1:
            raise ContractBuildError(
                'в данных несколько возможных колонок значения'
                'нужно явно указать target_source_col'
                f'Кандидаты: {candidate_value_cols}'
            )
        target_source_col = candidate_value_cols[0]
    else:
        if target_source_col not in candidate_value_cols:
            raise ContractBuildError(
                f'target_source_col={target_source_col!r} отсутствует в данных'
                f'Кандидаты: {candidate_value_cols}'
            )

    if exogenous_source_cols is None:
        exogenous_source_cols = [c for c in candidate_value_cols if c != target_source_col]
    else:
        unknown = [c for c in exogenous_source_cols if c not in candidate_value_cols]
        if unknown:
            raise ContractBuildError(f'exogenous_source_cols содержит неизвестные колонки: {unknown}')
        exogenous_source_cols = [c for c in exogenous_source_cols if c != target_source_col]

    # строим dataset (список строк)
    dataset_out: list[dict[str, Any]] = []
    for i, row in enumerate(iep_rows):
        if date_key not in row:
            raise ContractBuildError(f'iep_rows[{i}] не содержит ключа даты {date_key!r}')

        out_row: dict[str, Any] = {}
        try:
            out_row[timestamp_col] = parse_date_to_iso_z(row[date_key], field_name=f'iep_rows[{i}].{date_key}')
        except ParseError as e:
            raise ContractBuildError(str(e)) from e

        # таргерт
        if target_source_col not in row:
            raise ContractBuildError(f'iep_rows[{i}] не содержит target колонки {target_source_col!r}')
        try:
            out_row[target_col] = parse_float_like(row[target_source_col], field_name=f'iep_rows[{i}].{target_source_col}')
        except ParseError as e:
            raise ContractBuildError(str(e)) from e

        # экзогены
        for col in exogenous_source_cols:
            if col not in row:
                raise ContractBuildError(f'iep_rows[{i}] не содержит exogenous колонки {col!r}')
            try:
                out_row[col] = parse_float_like(row[col], field_name=f'iep_rows[{i}].{col}')
            except ParseError as e:
                raise ContractBuildError(str(e)) from e

        dataset_out.append(out_row)

    dataset_schema_out = {
        'timestamp_col': timestamp_col,
        'target_col': target_col,
        'series_id_col': series_id_col,
        'exogenous_cols': exogenous_source_cols,
    }

    ts_out = ts if ts is not None else _default_ts_config(horizon=horizon)
    training_out = training if training is not None else _default_training_config()

    payload: dict[str, Any] = {
        'dataset': dataset_out,
        'dataset_schema': dataset_schema_out,
        'ts': ts_out,
        'training': training_out,
        'idempotency_key': idempotency_key,
    }

    if endpoint == 'fit_auto':
        tuning_out = tuning if tuning is not None else _default_tuning_config()
        payload['tuning'] = tuning_out
    elif endpoint != 'fit':
        raise ContractBuildError(f'неизвестный endpoint: {endpoint!r}')

    return payload
