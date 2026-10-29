"""Sign-in. Each signed-in user gets a run store that carries their own credentials."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rag_eu_law.domain.exceptions import AuthenticationError
from rag_eu_law.domain.protocols import RunStore
from rag_eu_law.storage.supabase_store import SupabaseRunStore

logger = logging.getLogger(__name__)
SIGN_IN_FAILED = "Sign-in failed. Check your email and password."


@dataclass(frozen=True)
class AuthenticatedUser:
    email: str
    run_store: RunStore
    sign_out: Callable[[], None]


class Authenticator(Protocol):
    def sign_in(self, email: str, password: str) -> AuthenticatedUser:
        """Raise `AuthenticationError` when the credentials are not accepted."""
        ...


class SupabaseAuthenticator:
    """Email and password sign-in. Accounts are created by an admin in the Supabase dashboard."""

    def __init__(self, url: str, anon_key: str, client_factory: Callable[..., Any] | None = None):
        self._url = url
        self._anon_key = anon_key
        self._client_factory = client_factory or _default_client_factory

    def sign_in(self, email: str, password: str) -> AuthenticatedUser:
        client = self._client_factory(self._url, self._anon_key)
        try:
            response = client.auth.sign_in_with_password(
                {"email": email.strip(), "password": password}
            )
        except Exception as error:  # wrong password, unknown user, network: one message
            logger.warning("sign-in failed for %s: %s", email, error)
            raise AuthenticationError(SIGN_IN_FAILED) from error
        if response.user is None or response.session is None:
            raise AuthenticationError(SIGN_IN_FAILED)
        return AuthenticatedUser(
            email=str(response.user.email or email),
            run_store=SupabaseRunStore(client),
            sign_out=lambda: _sign_out(client),
        )


def _sign_out(client: Any) -> None:
    try:
        client.auth.sign_out()
    except Exception:  # signing out must never raise into the UI
        logger.warning("sign-out failed", exc_info=True)


def _default_client_factory(url: str, key: str) -> Any:
    from supabase import create_client  # noqa: PLC0415 - optional 'cloud' extra

    return create_client(url, key)
