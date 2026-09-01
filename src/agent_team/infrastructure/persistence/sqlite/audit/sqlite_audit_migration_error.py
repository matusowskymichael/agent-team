"""SQLite audit schema migration failure."""


class SQLiteAuditMigrationError(RuntimeError):
    """Raised when the local audit schema cannot be migrated."""
