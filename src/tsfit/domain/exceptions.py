from __future__ import annotations


class DomainError(Exception):
    '''Базовая ошибка домена'''


class ValidationError(DomainError):
    '''Ошибка валидации входных данных'''


class InvariantError(DomainError):
    '''Нарушение инварианта (жесткого правила)'''


class ConflictError(DomainError):
    '''Конфликт идемпотентности или состояния'''


class NotFoundError(DomainError):
    '''Не найден ресурс (например model_id)'''

class TrainingError(DomainError):
    '''Ошибка процесса обучения'''
