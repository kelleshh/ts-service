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


# метрики, где 'больше = лучше'
# все остальные считается 'меньше = лучше'
MAXIMIZE_METRICS_PREFIXES: tuple[str, ...] = (
    'auc',
    'aucpr',
    'pre',
    'ndcg',
    'map',
    'interval-regression-accuracy',
)


def is_higher_better(metric: str) -> bool:
    '''Возвращает True, если метрику надо МАКСИМИЗИРОВАТЬ.

    Доменное правило направления оптимизации метрики

    Правило:
    - auc/aucpr/pre/ndcg/map/interval-regression-accuracy — максимизируем
    - остальные — минимизируем

    Для вариантов с '@k' и/или '-' правило то же (по префиксу).
    '''

    if not isinstance(metric, str):
        return False

    m = metric.strip()
    if not m:
        return False

    # берем префикс до символов '@' или '-' (если '-' не часть имени вроде 'poisson-nloglik')
    # Для 'poisson-nloglik' префикс будет весь, но он не в списке maximize.
    name = m.split('@', 1)[0]

    # Для 'ndcg-' / 'map-' name будет 'ndcg-' -> поправим
    if name.endswith('-') and name in ('ndcg-', 'map-'):
        name = name[:-1]

    return name in MAXIMIZE_METRICS_PREFIXES