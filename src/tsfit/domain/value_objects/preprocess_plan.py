from __future__ import annotations

from dataclasses import dataclass


# Value Object
@dataclass(frozen=True, slots=True)
class PreprocessPlan:
    '''
    План препроцессинга, который потом воспроизводится в /predict
    Определяется в рамках политики обработки для конкретного датасета

    '''

    drop_missing_target: bool # удалять ли пропущенные таргеты
    ffill_exog: bool # делать ли forward fill для экзогенов
    add_missing_indicators: bool # добавлять ли индикаторы пропусков
