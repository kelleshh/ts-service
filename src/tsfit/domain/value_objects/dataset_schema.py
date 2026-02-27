from __future__ import annotations
from dataclasses import dataclass

from tsfit.domain.exceptions import ValidationError


# Value Object
@dataclass(frozen=True, slots=True)
class DatasetSchema:
    '''
    Описывает план-структуру входного датасета
    '''
    timestamp_col: str # кто является временем
    target_col: str # наш y
    exog_cols: tuple[str, ...] # экзогены

    def validate(self) -> None: # инварианты
        if not self.timestamp_col:
            raise ValidationError('timestamp_col пустой')
        if not self.target_col:
            raise ValidationError('target_col пустой')
        if self.target_col == self.timestamp_col:
            raise ValidationError('target_col совпадает с timestamp_col')
        if len(set(self.exog_cols)) != len(self.exog_cols):
            raise ValidationError('exog_cols содержит дубликаты')
        for c in self.exog_cols:
            if c in (self.timestamp_col, self.target_col):
                raise ValidationError('exog_cols пересекается с timestamp_col/target_col')

    @staticmethod
    def make(timestamp_col: str, target_col: str, exog_cols: tuple[str, ...] | None) -> 'DatasetSchema':
        '''
        Фабрика класса. Создает экземпляр и проверяет инварианты
        '''
        exog = tuple(exog_cols or [])
        obj = DatasetSchema(timestamp_col=timestamp_col, target_col=target_col, exog_cols=exog)
        obj.validate()
        return obj
