from __future__ import annotations

from dataclasses import dataclass

from tsfit.domain.exceptions import ValidationError


# Value Object
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    '''
    План как обучать модель (гперпараметры и правила обучения) в отрыве от конкретной реализации
    Фиксирует контракт обучения, который далее используется в /fit, /fit/auto
    '''
    primary_metric: str # какую метрику оптимизируем
    metrics: tuple[str, ...] # какие метрики считаем и лоигруем
    model_params: dict[str, float | int | str | bool] # гиперпараметры

    early_stopping_rounds: int # сколько итераций без улучшения терпим
    n_estimators_cap: int # потолок числа эстиматоров в ансамбле
    
    mape_eps: float = 1e-6 # эпсилон
    mape_zero_frac_threshold: float = 0.02 # порог для smape

    def validate(self) -> None: # проверка инвариантов
        if not self.primary_metric:
            raise ValidationError('primary_metric пустой')
        if self.primary_metric not in self.metrics:
            raise ValidationError('primary_metric должен входить в metrics')
        if self.early_stopping_rounds <= 0:
            raise ValidationError('early_stopping_rounds должен быть > 0')
        if self.n_estimators_cap <= 0:
            raise ValidationError('n_estimators_cap должен быть > 0')

    @staticmethod
    def make(
        primary_metric: str,
        metrics: tuple[str, ...],
        model_params: dict[str, float | int | str | bool] | None,
        early_stopping_rounds: int = 50,
        n_estimators_cap: int = 2000,
        mape_eps = 1e-6, # дефолты
        mape_zero_frac_threshold = 0.02, # дефлоты
        

    ) -> 'TrainingConfig':
        '''
        Фабрика класса. Проверяет доменные инварианты перед созданием экземпляра
        '''
        obj = TrainingConfig(
            primary_metric=primary_metric,
            metrics=tuple(metrics),
            model_params=dict(model_params or {}),
            early_stopping_rounds=early_stopping_rounds,
            n_estimators_cap=n_estimators_cap,
            mape_eps = mape_eps,
            mape_zero_frac_threshold = mape_zero_frac_threshold,
        )
        obj.validate()
        return obj
