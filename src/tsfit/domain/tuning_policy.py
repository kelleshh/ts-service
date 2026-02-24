from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetProfile:
    '''
    Профиль построенного датасета (статистики)
    набор чисел, по которым можно принимать решения в рамках оптимизации обучения
    '''
    n_rows_raw: int
    n_total_after_features: int
    n_train: int
    n_valid: int
    n_features: int


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    log: bool = False

@dataclass(frozen=True)
class TuningSearchSpace:
    '''
    Диапазоны для автоподбора параметров

    Описывает "какие значения разумно пробовать"
    Кроме learning rate (он подбирается отдельно в связке с n_trials)
    '''

    max_depth: IntRange
    subsample: FloatRange
    colsample_bytree: FloatRange
    min_child_weight: FloatRange
    reg_alpha: FloatRange
    reg_lambda: FloatRange


@dataclass(frozen=True)
class TuningPolicy:
    '''
    Доменная политика автоподбора

    В этой политике фиксируем правила:
    - как действовать на маленьких данных
    - какие разумные диапазоны пробовать
    - как ограничивать число попыток
    - как выбирать learning_rate и n_estimators как связку
    '''

    name: str
    data_bucket: str

    # Этап 1 - подбирается связка learning_rate + n_estimators
    learning_rate_grid: list[float]
    stage1_max_estimators: int
    stage1_early_stopping_rounds: int

    # Этап 2: подбираются остальные параметры
    n_trials_cap: int
    search_space: TuningSearchSpace



def choose_tuning_policy(profile: DatasetProfile) -> TuningPolicy:
    '''
    Выбирает политику автоподбора под профиль данных по эвристикам (пороги)
    '''

    n = int(profile.n_train)
    
    if n <= 150:
        data_bucket = 'small' # для отчета
        name = 'two_stage_small' # для отчета

        lr_grid = [0.03, 0.05, 0.1]
        stage1_max_estimators = 2000
        stage1_early_stopping_rounds = 30

        n_trials_cap = 15

        # узкие диапазоны сложности
        space = TuningSearchSpace(
            max_depth=IntRange(2, 5),
            subsample=FloatRange(0.7, 1.0),
            colsample_bytree=FloatRange(0.7, 1.0),
            min_child_weight=FloatRange(1.0, 20.0),
            reg_alpha=FloatRange(1e-4, 10.0, log=True),
            reg_lambda=FloatRange(1e-4, 30.0, log=True),
        )

    elif n <= 800:
        data_bucket = 'medium'
        name = 'two_stage_medium'

        lr_grid = [0.02, 0.05, 0.1, 0.2]
        stage1_max_estimators = 3000
        stage1_early_stopping_rounds = 50

        n_trials_cap = 25

        space = TuningSearchSpace(
            max_depth=IntRange(3, 8),
            subsample=FloatRange(0.6, 1.0),
            colsample_bytree=FloatRange(0.6, 1.0),
            min_child_weight=FloatRange(0.0, 15.0),
            reg_alpha=FloatRange(1e-6, 10.0, log=True),
            reg_lambda=FloatRange(1e-6, 20.0, log=True),
        )

    else:
        data_bucket = 'large'
        name = 'two_stage_large'

        lr_grid = [0.02, 0.05, 0.1, 0.2]
        stage1_max_estimators = 4000
        stage1_early_stopping_rounds = 80

        n_trials_cap = 35

        space = TuningSearchSpace(
            max_depth=IntRange(3, 10),
            subsample=FloatRange(0.5, 1.0),
            colsample_bytree=FloatRange(0.5, 1.0),
            min_child_weight=FloatRange(0.0, 10.0),
            reg_alpha=FloatRange(1e-8, 10.0, log=True),
            reg_lambda=FloatRange(1e-8, 15.0, log=True),
        )

    return TuningPolicy(
        name=name,
        data_bucket=data_bucket,
        learning_rate_grid=lr_grid,
        stage1_max_estimators=stage1_max_estimators,
        stage1_early_stopping_rounds=stage1_early_stopping_rounds,
        n_trials_cap=n_trials_cap,
        search_space=space,
    )