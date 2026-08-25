"""Database connection handling via SQLAlchemy + pyodbc.

Secrets are read from environment variables only, are never logged, and the
connection string is never printed in full.
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class ConnectionSettings:
    """Database connection configuration."""

    server: str
    database: str
    driver: str = "ODBC Driver 18 for SQL Server"
    auth_mode: str = "sql_password"
    username: str | None = None
    password: str | None = None
    encrypt: str = "yes"
    trust_server_certificate: str = "no"
    tenant_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if all required connection settings are present."""
        if not self.server or not self.database:
            return False
        if self.auth_mode == "sql_password":
            return bool(self.username and self.password)
        if self.auth_mode == "entra_service_principal":
            return bool(self.tenant_id and self.client_id and self.client_secret)
        return self.auth_mode == "entra_interactive"


def load_connection_settings(env_path: str = ".env") -> ConnectionSettings:
    """Load connection settings from environment variables."""
    load_dotenv(env_path, override=False)
    return ConnectionSettings(
        server=os.getenv("SQL_SERVER", ""),
        database=os.getenv("SQL_DATABASE", ""),
        driver=os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server"),
        auth_mode=os.getenv("SQL_AUTH_MODE", "sql_password"),
        username=os.getenv("SQL_USERNAME"),
        password=os.getenv("SQL_PASSWORD"),
        encrypt=os.getenv("SQL_ENCRYPT", "yes"),
        trust_server_certificate=os.getenv("SQL_TRUST_SERVER_CERTIFICATE", "no"),
        tenant_id=os.getenv("SQL_TENANT_ID"),
        client_id=os.getenv("SQL_CLIENT_ID"),
        client_secret=os.getenv("SQL_CLIENT_SECRET"),
    )


def _build_connection_string(settings: ConnectionSettings) -> str:
    """Build ODBC connection string from settings."""
    encrypt = "yes" if settings.encrypt.lower() in ("yes", "true", "1") else "no"
    trust = "yes" if settings.trust_server_certificate.lower() in ("yes", "true", "1") else "no"

    base = (
        f"Driver={{{settings.driver}}};"
        f"Server={settings.server};"
        f"Database={settings.database};"
        f"Encrypt={encrypt};TrustServerCertificate={trust};"
    )
    if settings.auth_mode == "sql_password":
        base += f"UID={settings.username};PWD={settings.password};"
    elif settings.auth_mode == "entra_interactive":
        base += "Authentication=ActiveDirectoryInteractive;"
    elif settings.auth_mode == "entra_service_principal":
        base += (
            "Authentication=ActiveDirectoryServicePrincipal;"
            f"UID={settings.client_id};PWD={settings.client_secret};"
        )
    else:
        raise ValueError(f"Unsupported SQL_AUTH_MODE: {settings.auth_mode}")
    return base


def _redact(value: Any) -> str:
    """Redact sensitive values for logging."""
    if value is None or value == "":
        return "<empty>"
    return "<redacted>"


def redacted_connection_summary(settings: ConnectionSettings) -> dict[str, str]:
    """Return a dict safe to log with secrets redacted."""
    return {
        "server": settings.server or "<not set>",
        "database": settings.database or "<not set>",
        "driver": settings.driver,
        "auth_mode": settings.auth_mode,
        "username": settings.username or "<not set>",
        "password": _redact(settings.password),
        "client_secret": _redact(settings.client_secret),
    }


def get_engine(settings: ConnectionSettings | None = None) -> Engine:
    """Create SQLAlchemy engine from connection settings."""
    if settings is None:
        settings = load_connection_settings()
    odbc_str = _build_connection_string(settings)
    params = urllib.parse.quote_plus(odbc_str)
    url = f"mssql+pyodbc:///?odbc_connect={params}"
    return create_engine(url, pool_pre_ping=True)


def test_connection(engine: Engine) -> dict[str, Any]:
    """Test database connection and return version info."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT @@VERSION AS version, DB_NAME() AS db_name")
            ).mappings().first()
            if row is None:
                return {"connected": False, "error": "No result from version query"}
            return {
                "connected": True,
                "server_version": row["version"],
                "database": row["db_name"],
            }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def determine_mode(requested_mode: str = "auto") -> tuple[str, dict[str, Any]]:
    """Determine whether to use connected or offline mode.

    Returns (mode, details) where mode is 'connected' or 'offline'.
    """
    settings = load_connection_settings()
    details: dict[str, Any] = {"connection_summary": redacted_connection_summary(settings)}

    if requested_mode == "offline":
        details["reason"] = "Offline mode explicitly requested."
        return "offline", details

    if not settings.is_complete:
        details["reason"] = "Connection settings incomplete in .env (or .env not present)."
        if requested_mode == "connected":
            details["error"] = "Connected mode requested but credentials incomplete."
        return "offline", details

    try:
        engine = get_engine(settings)
        test = test_connection(engine)
        details["connection_test"] = test
        if test.get("connected"):
            return "connected", details
        details["reason"] = f"Connection test failed: {test.get('error')}"
        return "offline", details
    except Exception as e:
        details["reason"] = f"Could not build engine / driver unavailable: {e}"
        return "offline", details
