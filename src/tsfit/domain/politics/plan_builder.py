from __future__ import annotations

from dataclasses import dataclass
import math

from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import CVPlan, FeaturePlan, PreprocessPlan, SeriesProfile


# Policy
@dataclass(frozen=True, slots=True)
class PlanBuilder:
    '''
    Политика выбора фичей и схемы walk-forward.
    - определяет FeaturePlan, CVPlan, PreprocessPlan
    - имеет право уменьшать max_lag, чтобы обеспечить >= 2 фолда на кросс-валидации
    '''

    policy_version: str = '2026-02-26' # ну не 30 февраля главное

    def build_preprocess_plan(self) -> PreprocessPlan:
        return PreprocessPlan(
            drop_missing_target=True, # базово в целях безопасности НЕ ОБУЧАЕМ НИЧЕГО НА СТРОКАХ ГДЕ y =nan
            ffill_exog=True,
            add_missing_indicators=True,
        )

    def build_feature_plan(self, profile: SeriesProfile) -> FeaturePlan:
        n = profile.n_train

        if n < 10:
            raise ValidationError('Слишком мало данных после препроцессинга')

        if n < 120: # маленький датасет
            max_lag = max(1, min(12, n // 8))
            rolling = (3,)
            rolling_std = ()

        elif n < 1000: # средний датасет
            max_lag = max(1, min(30, n // 10))
            rolling = (3, 7, 14)
            rolling_std = (7, 14)

        else: # большой датасет
            max_lag = max(1, min(60, n // 20))
            rolling = (3, 7, 14, 30) # от балды (НАЗЫВАЕТСЯ ЭВРИСТИКА, для упрощения) TODO: если не работает поменять нафиг
            rolling_std = (7, 14, 30)

        rolling = tuple(w for w in rolling if 1 <= w <= max_lag)
        rolling_std = tuple(w for w in rolling_std if 1 <= w <= max_lag)

        p = profile.season_period
        add_season = (p is not None) and (int(p) > 1)

        return FeaturePlan(
            max_lag=max_lag,
            rolling_windows=rolling,
            rolling_std_windows=rolling_std,
            add_delta_t=not profile.is_regular,
            add_time_index=True,
            season_period=int(p) if add_season else None, #type: ignore
            add_season_sin_cos=add_season,
            add_exp_features=profile.is_exponential,
        )

    def build_cv_plan(self, n_train: int, horizon: int, feature_plan: FeaturePlan) -> CVPlan: # он строится еще и на фичеплане
        n = int(n_train)
        h = int(horizon)
        if h < 1:
            raise ValidationError('horizon должен быть >= 1')
        if n < 10:
            raise ValidationError('Слишком мало данных для CV')

        min_train_size = max(1, int(math.ceil(0.15 * n_train)))
        valid_size = max(int(math.ceil(0.15 * n_train)), feature_plan.max_lag + horizon + 1)

        if n <= min_train_size + valid_size:
            raise ValidationError('insufficient_cv_folds')

        k_max = (n - min_train_size) // valid_size
        k_max = int(k_max)
        if k_max < 2:
            raise ValidationError('insufficient_cv_folds')

        if n < 120:
            desired_max = 3
        elif n < 1000:
            desired_max = 5
        else:
            desired_max = 8

        k = min(k_max, desired_max)
        if k < 2:
            raise ValidationError('insufficient_cv_folds')

        return CVPlan(k_folds=k, min_train_size=min_train_size, valid_size=valid_size)

    def build_all(self, profile: SeriesProfile, horizon: int) -> tuple[PreprocessPlan, FeaturePlan, CVPlan]:
        pp = self.build_preprocess_plan()
        fp = self.build_feature_plan(profile)

        # shrink max_lag, чтобы гарантировать >= 2 фолда
        while True:
            try:
                cv = self.build_cv_plan(profile.n_train, horizon, fp)
                return pp, fp, cv
            except ValidationError as e:
                if str(e) != 'insufficient_cv_folds':
                    raise
                if fp.max_lag <= 1:
                    raise ValidationError('Слишком мало данных для walk-forward CV (нужно хотя бы 2 фолда)')
                new_max = fp.max_lag - 1
                fp = FeaturePlan(
                    max_lag=new_max,
                    rolling_windows=tuple(w for w in fp.rolling_windows if w <= new_max),
                    rolling_std_windows=tuple(w for w in fp.rolling_std_windows if w <= new_max),
                    add_delta_t=fp.add_delta_t,
                    add_time_index=fp.add_time_index,
                    season_period=fp.season_period,
                    add_season_sin_cos=fp.add_season_sin_cos,
                    add_exp_features=fp.add_exp_features,
                )
