from __future__ import annotations

import pandas as pd

from tsfit.application.ports import WalkForwardSplitter
from tsfit.application.types import DatasetHandle, Fold
from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import CVPlan


# Adapter (Infrastructure)
class DefaultWalkForwardSplitter(WalkForwardSplitter):
    def split(self, supervised: DatasetHandle, plan: CVPlan) -> list[Fold]:
        df = supervised.payload
        if not isinstance(df, pd.DataFrame):
            raise ValidationError('supervised должен быть DataFrame внутри DatasetHandle')

        n = int(len(df))
        if n <= 0:
            raise ValidationError('пустой supervised')
        if plan.k_folds < 2:
            raise ValidationError('k_folds должен быть >= 2')

        folds: list[Fold] = []
        for i in range(plan.k_folds):
            train_end = plan.min_train_size + i * plan.valid_size
            if train_end >= n:
                break
            valid_end = train_end + plan.valid_size
            if i == plan.k_folds - 1:
                valid_end = n
            if valid_end > n:
                valid_end = n
            if valid_end <= train_end:
                break

            train_df = df.iloc[:train_end].copy()
            valid_df = df.iloc[train_end:valid_end].copy()

            # переносим attrs
            train_df.attrs = dict(df.attrs)
            valid_df.attrs = dict(df.attrs)

            folds.append(Fold(train=DatasetHandle(train_df), valid=DatasetHandle(valid_df)))

        if len(folds) < 2:
            raise ValidationError('не получилось собрать >= 2 фолда')
        return folds
