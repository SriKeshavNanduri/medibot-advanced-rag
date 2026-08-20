"""JWT issuing and verification (PyJWT 2.13) plus the FastAPI auth dependency."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Annotated

import jwt  # PyJWT
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from api import db
from api.config import get_settings
from api.rbac import Role

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(
    scheme_name="MediBotJWT",
    description="Bearer token issued by POST /login.",
    auto_error=True,
)

# Constant-time decoy so an unknown username costs the same as a wrong password.
_DECOY_HASH, _DECOY_SALT = db.hash_password(uuid.uuid4().hex)


class CurrentUser(BaseModel):
    username: str
    role: Role
    jti: str = ""

    def session_key(self, client_session_id: str) -> str:
        """
        Generates a namespaced session key using the authenticated username and a client-provided session ID.
        This prevents one user from accessing another user's conversation history.

        Args:
            client_session_id (str): The session ID provided by the client.

        Returns:
            str: A unique session key combining the username and client session ID.
        """
        """Namespace the client's session id with the authenticated username.

        Without this, any user could read another user's conversation simply by
        guessing their session id.
        """
        return f"{self.username}::{client_session_id}"


def authenticate(username: str, password: str) -> Role | None:
    """
    Authenticates a user against the `app_users` table.

    Args:
        username (str): The username to authenticate.
        password (str): The password to verify.

    Returns:
        Role | None: The `Role` of the authenticated user if successful,
                      otherwise `None` if authentication fails.
    Verify credentials against the app_users table. None on any failure.
    """
    record = db.get_user(username)
    if record is None:
        db.verify_password(password, _DECOY_HASH, _DECOY_SALT)  # equalise timing
        return None
    if not db.verify_password(password, record.password_hash, record.password_salt):
        return None
    try:
        return Role(record.role)
    except ValueError:
        logger.error("User %s has unknown role %r in the database", username, record.role)
        return None


def create_access_token(username: str, role: Role) -> tuple[str, int]:
    """
    Creates a JSON Web Token (JWT) for an authenticated user.

    Args:
        username (str): The username for whom the token is being created.
        role (Role): The role of the user, which will be embedded in the token claims.
    Returns:
        tuple[str, int]: A tuple containing the encoded JWT token string and its time-to-live in seconds.
    """
    settings = get_settings()
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role.value,  # the ONLY source of truth for RBAC
        "scope": "medibot:chat",
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "nbf": now,
        "exp": now + settings.jwt_ttl_seconds,
        "jti": uuid.uuid4().hex,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, settings.jwt_ttl_seconds


def _unauthorized(detail: str) -> HTTPException:
    """
    Helper function to raise a 401 Unauthorized HTTPException.

    Args:
        detail (str): A message providing more details about the unauthorized access.
    Returns:
        HTTPException: An HTTPException configured for 401 Unauthorized.
    """
def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
) -> CurrentUser:
    """
    FastAPI dependency that decodes and validates a JWT token from the Authorization header.
    
    Args:
        credentials (Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]):
            The bearer token credentials extracted from the request header.
    
    Returns:
        CurrentUser: An object containing the username, role, and JTI from the validated token.
    
    Raises:
        HTTPException: If the token is missing, expired, invalid, or contains an unknown role,
                       raising a 401 Unauthorized or 403 Forbidden error.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],  # explicit: blocks alg=none confusion
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token has expired; call POST /login again.")
    except jwt.InvalidTokenError as exc:
        raise _unauthorized(f"Invalid token: {exc}")

    try:
        role = Role(payload["role"])
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Token carries an unknown role: {payload.get('role')!r}",
        )

    return CurrentUser(username=payload["sub"], role=role, jti=payload.get("jti", ""))


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]
