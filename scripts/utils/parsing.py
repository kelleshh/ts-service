from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class ParseError(Exception):
    '''
    Ошибка парсинга в контракт
    '''
    message: str

    def __str__(self) -> str:
        return self.message


def parse_float_like(value: Any, *, field_name: str) -> float:
    '''
    Преобразует значение в float: строки как с точкой так и с запятой
    '''
    if value is None:
        raise ParseError(f'{field_name}: значение отсутствует (null)')

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ParseError(f'{field_name}: пустая строка')

        s = s.replace(' ', '')
        # поддержка десятичной запятой
        if s.count(',') == 1 and s.count('.') == 0:
            s = s.replace(',', '.')

        try:
            return float(s)
        except ValueError as e:
            raise ParseError(f'{field_name}: не удалось преобразовать в число: {value!r}') from e

    raise ParseError(f'{field_name}: неподдерживаемый тип {type(value)}')


def parse_date_to_iso_z(value: Any, *, field_name: str) -> str:
    '''
    Преобразует дату/время к строке ISO формата в UTC, с суффиксом 'Z'.

    преобразует
    - datetime (naive или tz-aware)
    - строки 'DD.MM.YYYY', 'YYYY-MM-DD' ISO ('2024-01-01T00:00:00Z' или с '+00:00')
    '''
    if value is None:
        raise ParseError(f'{field_name}: значение отсутствует (null)')

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            raise ParseError(f'{field_name}: пустая строка')

        # 1) DD.MM.YYYY
        try:
            dt = datetime.strptime(s, '%d.%m.%Y')
        except ValueError:
            dt = None  # type: ignore

        # 2) YYYY-MM-DD
        if dt is None:
            try:
                dt = datetime.strptime(s, '%Y-%m-%d')
            except ValueError:
                dt = None  # type: ignore

        # 3) ISO
        if dt is None:
            s_iso = s.replace('Z', '+00:00')
            try:
                dt = datetime.fromisoformat(s_iso)
            except ValueError as e:
                raise ParseError(f'{field_name}: не удалось распарсить дату: {value!r}') from e
    else:
        raise ParseError(f'{field_name}: неподдерживаемый тип {type(value)}')

    # нормализуем в UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
