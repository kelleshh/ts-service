from __future__ import annotations

from typing import Any

from tsfit.application.ports import BuiltDataset, HyperparameterTuner
from tsfit.domain.tuning_policy import DatasetProfile, choose_tuning_policy
from tsfit.domain.value_objects import TrainingConfigValueObject, TuningConfigValueObject

from tsfit.infrastructure.training.xgb_defaults import merge_params
from tsfit.infrastructure.training.optuna.search_space import policy_debug_dict
from tsfit.infrastructure.training.optuna.stage1_lr_n_estimators import pick_learning_rate_and_n_estimators
from tsfit.infrastructure.training.optuna.stage2_optuna_search import run_optuna_search


class OptunaXGBTuner(HyperparameterTuner):
    '''
    Инфраструктурная реализация автоподбора через optuna.

    Исполняется выбранная доменом политика обучения (этапы диапазоны) с помощью optuna и xgboost
    '''

    def tune(
        self,
        dataset: BuiltDataset,
        training: TrainingConfigValueObject,
        tuning: TuningConfigValueObject,
    ) -> tuple[TrainingConfigValueObject, dict[str, Any]]:
        profile = DatasetProfile(
            n_rows_raw=int(dataset.n_rows_raw),
            n_total_after_features=int(dataset.n_total_after_features),
            n_train=int(dataset.n_train),
            n_valid=int(dataset.n_valid),
            n_features=int(dataset.n_features),
        )

        policy = choose_tuning_policy(profile)

        base_params = merge_params(training.model_params)

        fixed_params, stage1_report = pick_learning_rate_and_n_estimators(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_valid=dataset.X_valid,
            y_valid=dataset.y_valid,
            base_params=base_params,
            primary_metric=training.primary_metric,
            learning_rate_grid=policy.learning_rate_grid,
            max_estimators=policy.stage1_max_estimators,
            early_stopping_rounds=policy.stage1_early_stopping_rounds,
        )

        best_params, stage2_report = run_optuna_search(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_valid=dataset.X_valid,
            y_valid=dataset.y_valid,
            base_params=base_params,
            fixed_params=fixed_params,
            primary_metric=training.primary_metric,
            profile=profile,
            policy=policy,
            n_trials_requested=tuning.n_trials,
            timeout_sec=tuning.timeout_sec,
        )

        final_params = dict(base_params)
        final_params.update(fixed_params)
        final_params.update(best_params)

        final_training = TrainingConfigValueObject(
            model_params=final_params,
            metrics=training.metrics,
            primary_metric=training.primary_metric,
        )

        report = {
            'strategy_used': str(policy.name),
            'data_bucket': str(policy.data_bucket),
            'policy': policy_debug_dict(policy),
            'data_profile': {
                'n_rows_raw': profile.n_rows_raw,
                'n_total_after_features': profile.n_total_after_features,
                'n_train': profile.n_train,
                'n_valid': profile.n_valid,
                'n_features': profile.n_features,
            },
            'stage1': stage1_report,
            'stage2': stage2_report,
        }

        return final_training, report
