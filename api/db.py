"""Postgres-backed user store for /login.

Uses psycopg v3 (the same driver ``PostgresChatMessageHistory`` requires) and
stdlib PBKDF2 for password hashing -- neither ``passlib`` nor ``python-jose``
is installed in this environment, and neither is needed.
"""

from __future__ import annotations

import binascii
import hashlib
import hmac
import logging
import os
from contextlib import contextmanager
from typing import Iterator, NamedTuple

import psycopg
from psycopg import sql

from api.config import get_settings

logger = logging.getLogger(__name__)

PBKDF2_ROUNDS = 200_000
SALT_BYTES = 16


class UserRecord(NamedTuple):
    username: str
    password_hash: str
    password_salt: str
    role: str


# ------------------------------------------------------------- hashing ---
def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """
    Hashes a password using PBKDF2 with SHA256.
    
    Args:
        password (str): The plaintext password to hash.
        salt (bytes | None): An optional salt. If None, a new random salt is generated.
    Returns:
        tuple[str, str]: A tuple containing the hexadecimal representation of the hash and the salt.
    """
    """Return ``(hex_hash, hex_salt)``. Generates a fresh salt when omitted."""
    salt = salt if salt is not None else os.urandom(SALT_BYTES)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return binascii.hexlify(derived).decode(), binascii.hexlify(salt).decode()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """
    Verifies a plaintext password against a stored hash and salt.
    
    Args:
        password (str): The plaintext password to verify.
        stored_hash (str): The hexadecimal string of the stored password hash.
        stored_salt (str): The hexadecimal string of the stored salt.
    Returns:
        bool: True if the password matches, False otherwise.
    """
    try:
        salt = binascii.unhexlify(stored_salt)
    except (binascii.Error, ValueError):
        return False
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ---------------------------------------------------------- connections ---
@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """
    Provides a context-managed connection to the PostgreSQL database.
    The connection is committed on successful exit and rolled back on exceptions.
    
    Yields:
        psycopg.Connection: A database connection object.
    """
    """Short-lived psycopg v3 connection, committed on clean exit."""
    with psycopg.connect(get_settings().postgres_dsn, connect_timeout=10) as conn:
        yield conn


# --------------------------------------------------------------- schema ---
def ensure_users_table() -> None:
    """
    Ensures that the `app_users` table exists in the database.
    If the table does not exist, it is created.
    """
    table = sql.Identifier(get_settings().users_table)
    statement = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {} (
            username      TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            role          TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    ).format(table)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
        conn.commit()


def upsert_user(username: str, password: str, role: str, *, overwrite: bool = False) -> bool:
    """
    Inserts a new user or updates an existing one in the `app_users` table.
    
    Args:
        username (str): The username of the user.
        password (str): The plaintext password of the user.
        role (str): The role of the user.
        overwrite (bool): If True, updates existing user's password and role. If False, does nothing if user exists.
    Returns:
        bool: True if a row was inserted or updated, False otherwise.
    """
    """Insert a user. Returns True when a row was written.

    With ``overwrite=False`` an existing username is left untouched, so re-running
    the seed script never rotates someone's password behind their back.
    """
    password_hash, password_salt = hash_password(password)
    table = sql.Identifier(get_settings().users_table)
    conflict = (
        sql.SQL(
            "DO UPDATE SET password_hash = EXCLUDED.password_hash, "
            "password_salt = EXCLUDED.password_salt, role = EXCLUDED.role"
        )
        if overwrite
        else sql.SQL("DO NOTHING")
    )
    statement = sql.SQL(
        "INSERT INTO {} (username, password_hash, password_salt, role) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (username) {}"
    ).format(table, conflict)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, (username, password_hash, password_salt, role))
            written = cur.rowcount > 0
        conn.commit()
    return written


def create_user(username: str, password: str, role: str = "doctor") -> bool:
    """
    Create a user using the existing PBKDF2 password hashing flow.

    This preserves the current encryption and database logic; it simply wraps the
    lower-level insert helper with a clearer username/password API.
    """
    ensure_users_table()
    return upsert_user(username, password, role, overwrite=False)


def get_user(username: str) -> UserRecord | None:
    """
    Retrieves a user record from the `app_users` table by username.
    
    Args:
        username (str): The username of the user to retrieve.
    Returns:
        UserRecord | None: A `UserRecord` named tuple if the user is found,
                           otherwise `None`.
    """
    table = sql.Identifier(get_settings().users_table)
    statement = sql.SQL(
        "SELECT username, password_hash, password_salt, role FROM {} WHERE username = %s"
    ).format(table)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, (username,))
            row = cur.fetchone()
    return UserRecord(*row) if row else None


def count_users() -> int:
    """
    Counts the total number of users in the `app_users` table.
    
    Returns:
        int: The number of user records in the table.
    """
    table = sql.Identifier(get_settings().users_table)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(table))
            return int(cur.fetchone()[0])
