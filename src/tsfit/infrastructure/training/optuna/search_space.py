from __future__ import annotations

from dataclasses import asdict

import optuna

from tsfit.domain.tuning_policy import DatasetProfile, TuningPolicy


def suggest_params(trial: optuna.Trial, policy: TuningPolicy, profile: DatasetProfile) -> dict[str, float | int]:
    '''
    Генерирует набор параметров для этапа 2

    - learning_rate и n_estimators уже считаются выбранными на этапе 1
    - диапазоны берем из доменной политики (TuningPolicy)
    '''

    space = policy.search_space

    # доп ограничитель: если признаков мало, не даем слишком большую глубину чтобы избежать оверфита
    max_depth_high = int(space.max_depth.high)
    if profile.n_features <= 8:
        max_depth_high = min(max_depth_high, 6)
    if profile.n_features <= 4:
        max_depth_high = min(max_depth_high, 4)

    max_depth_low = int(space.max_depth.low)
    if max_depth_low > max_depth_high:
        max_depth_low = max_depth_high


    out: dict[str, float | int] = {}

    out['max_depth']          = trial.suggest_int('max_depth',
                                                  max_depth_low,
                                                  max_depth_high)
    out['subsample']        = trial.suggest_float('subsample',
                                                  float(space.subsample.low),
                                                  float(space.subsample.high))
    out['colsample_bytree'] = trial.suggest_float('colsample_bytree',
                                                  float(space.colsample_bytree.low),
                                                  float(space.colsample_bytree.high))
    out['min_child_weight'] = trial.suggest_float('min_child_weight',
                                                  float(space.min_child_weight.low),
                                                  float(space.min_child_weight.high))
    if space.reg_alpha.log:
        out['reg_alpha']    = trial.suggest_float('reg_alpha',
                                                  float(space.reg_alpha.low),
                                                  float(space.reg_alpha.high),
                                                  log=True)
    else:
        out['reg_alpha']    = trial.suggest_float('reg_alpha',
                                                  float(space.reg_alpha.low),
                                                  float(space.reg_alpha.high))
    if space.reg_lambda.log:
        out['reg_lambda']   = trial.suggest_float('reg_lambda',
                                                  float(space.reg_lambda.low),
                                                  float(space.reg_lambda.high),
                                                  log=True)
    else:
        out['reg_lambda']   = trial.suggest_float('reg_lambda',
                                                  float(space.reg_lambda.low),
                                                  float(space.reg_lambda.high))

    return out


def policy_debug_dict(policy: TuningPolicy) -> dict[str, object]:
    d = asdict(policy)
    # asdict раскроет датаклассы диапазонов
    return d
