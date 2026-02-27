from __future__ import annotations

import json
import os
import tempfile

from tsfit.domain.value_objects.meta import ModelMeta
from tsfit.application.ports import ModelRegistry
from tsfit.application.results import TrainingReport
from tsfit.domain.exceptions import NotFoundError


# Adapter (Infrastructure)
class MlflowModelRegistry(ModelRegistry):
    def __init__(
        self,
        *,
        experiment_name: str = 'tsfit',
        artifact_path: str = 'model',
    ) -> None:
        try:
            import mlflow
        except Exception as e:  # pragma: no cover
            raise RuntimeError('mlflow не установлен') from e

        self._mlflow = mlflow
        self._artifact_path = artifact_path

        self._mlflow.set_experiment(experiment_name)

    def find_by_idempotency_key(self, idempotency_key: str) -> tuple[str, str] | None:
        runs = self._mlflow.search_runs(
            filter_string=f"tags.idempotency_key = '{idempotency_key}'",
            output_format='pandas',
        )
        if runs is None or len(runs) == 0:
            return None

        run_id = str(runs.iloc[0]['run_id'])
        payload_hash = str(runs.iloc[0].get('tags.payload_hash', ''))
        return run_id, payload_hash

    def save(
        self,
        *,
        idempotency_key: str,
        payload_hash: str,
        report: TrainingReport,
        meta: ModelMeta,
        summary: dict,
        extra: dict[str, object],
    ) -> str:
        import xgboost as xgb

        if not isinstance(report.model, xgb.Booster):
            raise RuntimeError('ожидался xgboost.Booster для сохранения в mlflow')

        with self._mlflow.start_run() as run:
            run_id = run.info.run_id

            self._mlflow.set_tag('idempotency_key', idempotency_key)
            self._mlflow.set_tag('payload_hash', payload_hash)
            self._mlflow.set_tag('policy_version', meta.policy_version)
            self._mlflow.log_params({
                **{f'model_param.{k}': v for k, v in report.model_params.items()},
                'horizon': meta.horizon,
                'primary_metric': meta.training_config.primary_metric,
            })

            self._mlflow.xgboost.log_model(report.model, artifact_path=self._artifact_path)

            with tempfile.TemporaryDirectory() as tmp:
                meta_path = os.path.join(tmp, 'meta.json')
                summary_path = os.path.join(tmp, 'summary.json')
                extra_path = os.path.join(tmp, 'extra.json')

                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                with open(extra_path, 'w', encoding='utf-8') as f:
                    json.dump(extra, f, ensure_ascii=False, indent=2, default=str)

                self._mlflow.log_artifact(meta_path, artifact_path='artifacts')
                self._mlflow.log_artifact(summary_path, artifact_path='artifacts')
                self._mlflow.log_artifact(extra_path, artifact_path='artifacts')

            return str(run_id)

    def load_model_and_meta(self, model_id: str) -> tuple[object, ModelMeta]:
        import xgboost as xgb

        try:
            model = self._mlflow.xgboost.load_model(f'runs:/{model_id}/{self._artifact_path}')
        except Exception as e:  # pragma: no cover
            raise NotFoundError('model_id не найден в mlflow') from e

        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = self._mlflow.artifacts.download_artifacts(
                    run_id=model_id,
                    artifact_path='artifacts/meta.json',
                    dst_path=tmp,
                )
            except Exception as e:  # pragma: no cover
                raise NotFoundError('meta.json не найден') from e

            with open(path, 'r', encoding='utf-8') as f:
                meta_d = json.load(f)

        return model, ModelMeta.from_dict(meta_d)

    def load_summary(self, model_id: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = self._mlflow.artifacts.download_artifacts(
                    run_id=model_id,
                    artifact_path='artifacts/summary.json',
                    dst_path=tmp,
                )
            except Exception as e:  # pragma: no cover
                raise NotFoundError('summary.json не найден') from e

            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
