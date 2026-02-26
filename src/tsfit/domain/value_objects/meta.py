from __future__ import annotations

from dataclasses import asdict, dataclass

from tsfit.domain.value_objects import CVPlan, DatasetSchema, FeaturePlan, PreprocessPlan, TrainingConfig


# Value Object
@dataclass(frozen=True, slots=True)
class ModelMeta:
    '''
    Доменный снапшот конфигурации модели для воспроизводимости на инференсе, аудита обучения и т.д.
    '''
    schema: DatasetSchema # контракт данных
    preprocess_plan: PreprocessPlan # план очистки данных (будет воспроизводиться в предикте)
    feature_plan: FeaturePlan # план фичегена
    cv_plan: CVPlan # план кросс валидации
    horizon: int # горизонт прогноза
    policy_version: str # версия доменной политики
    training_config: TrainingConfig # конфигурация обучения
    feature_names: tuple[str, ...] # порядок фичей

    def to_dict(self) -> dict:
        '''
        Сериализация в примитивный словарь для хранения например на MLFlow
        '''
        return {
            'schema': {
                'timestamp_col': self.schema.timestamp_col,
                'target_col': self.schema.target_col,
                'exog_cols': list(self.schema.exog_cols),
            },
            'preprocess_plan': asdict(self.preprocess_plan),
            'feature_plan': asdict(self.feature_plan),
            'cv_plan': asdict(self.cv_plan),
            'horizon': self.horizon,
            'policy_version': self.policy_version,
            'training_config': {
                'primary_metric': self.training_config.primary_metric,
                'metrics': list(self.training_config.metrics),
                'model_params': dict(self.training_config.model_params),
                'early_stopping_rounds': self.training_config.early_stopping_rounds,
                'n_estimators_cap': self.training_config.n_estimators_cap,
                'mape_eps': self.training_config.mape_eps,
                'mape_zero_frac_threshold': self.training_config.mape_zero_frac_threshold,
            },
            'feature_names': list(self.feature_names),
        }

    @staticmethod
    def from_dict(d: dict) -> 'ModelMeta':
        '''
        Восстановление доменного обьекта из сериализованного представления
        Повторно прогоняет инварианты через фабрики
        '''
        schema_d = d['schema']
        schema = DatasetSchema.make(schema_d['timestamp_col'], schema_d['target_col'], schema_d.get('exog_cols'))
        pp = PreprocessPlan(**d['preprocess_plan'])
        fp = FeaturePlan(**d['feature_plan'])
        cv = CVPlan(**d['cv_plan'])
        tc_d = d['training_config']
        tc = TrainingConfig.make(
            primary_metric=tc_d['primary_metric'],
            metrics=tuple(tc_d['metrics']),
            model_params=dict(tc_d.get('model_params') or {}),
            early_stopping_rounds=int(tc_d['early_stopping_rounds']),
            n_estimators_cap=int(tc_d['n_estimators_cap']),
            mape_eps=float(tc_d['mape_eps']),
            mape_zero_frac_threshold=float(tc_d['mape_zero_frac_threshold']),
        )
        return ModelMeta(
            schema=schema,
            preprocess_plan=pp,
            feature_plan=fp,
            cv_plan=cv,
            horizon=int(d['horizon']),
            policy_version=str(d['policy_version']),
            training_config=tc,
            feature_names=tuple(d['feature_names']),
        )
