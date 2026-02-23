from tsfit.domain.exceptions import ValidationError
from tsfit.domain.value_objects import DatasetSchemaValueObject


def rule_validate_rows_have_columns(rows: list[dict], schema: DatasetSchemaValueObject) -> None:
    if not rows:
        raise ValidationError('dataset пустой')

    required = {schema.timestamp_col, schema.target_col, *schema.exogenous_cols}
    if schema.series_id_col:
        required.add(schema.series_id_col)

    for i, r in enumerate(rows):
        missing = required - set(r.keys())
        if missing:
            raise ValidationError(f'в row[{i}] нет колонок: {sorted(missing)}')