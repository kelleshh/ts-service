from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import DatasetSchemaValueObject


def rule_validate_rows_have_columns(rows: list[dict], schema: DatasetSchemaValueObject) -> None:
    '''
    Проверка структурной целостности:
    - датасет не пустой
    - каждая строка содержит timestamp_col, target_col и все exogenous_cols
    - если указан series_id_col, то он тоже должен быть
    '''
    
    if not rows:
        raise ValidationError('dataset пустой')

    required = {schema.timestamp_col, schema.target_col, *schema.exogenous_cols}
    if schema.series_id_col:
        required.add(schema.series_id_col)

    for i, r in enumerate(rows):
        missing = required - set(r.keys())
        if missing:
            raise ValidationError(f'в row[{i}] нет колонок: {sorted(missing)}')
        
# доменные базовые метрики xgboost (инвариант)

BASE_XGBOOST_EVAL_METRICS: frozenset[str] = frozenset(
    {
        'rmse',
        'rmsle',
        'mae',
        'mape',
        'mphe',
        'logloss',
        'error',
        'merror',
        'mlogloss',
        'auc',
        'aucpr',
        'pre',
        'ndcg',
        'map',
        'poisson-nloglik',
        'gamma-nloglik',
        'cox-nloglik',
        'gamma-deviance',
        'tweedie-nloglik',
        'aft-nloglik',
        'interval-regression-accuracy',
    }
)