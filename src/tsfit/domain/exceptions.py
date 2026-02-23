class DomainError(Exception):
    '''
    Ошибка уровня доменных правил (инварианты)
    '''
    pass

class ValidationError(DomainError):
    '''
    Некорректный вход
    '''
    pass

class IdempotencyConflict(DomainError):
    '''
    Один и тот же ключ идемпотентности, но разные данные
    '''
    pass

class InvalidRunTransition(DomainError):
    '''
    Недопустимый переход статуса запуска
    '''
    pass

class TrainingFailed(DomainError):
    '''
    Обучение упало по внутренней причине
    '''
    pass