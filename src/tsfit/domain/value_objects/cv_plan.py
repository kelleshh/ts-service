from __future__ import annotations
from dataclasses import dataclass


# Value Object
@dataclass(frozen=True, slots=True)
class CVPlan:
    '''
    План разрезания supervized-датасета на фолды в walk-forward кросс-валидации
    '''
    k_folds: int # сколько шагов walk-forward
    min_train_size: int # минимальный train в фолде
    valid_size: int # размер валидационной выборки
