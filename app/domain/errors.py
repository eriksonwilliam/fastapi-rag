class DomainError(Exception):
    """Erro base do dominio."""


class ValidationError(DomainError):
    """Entrada invalida (mapeada para HTTP 400)."""
