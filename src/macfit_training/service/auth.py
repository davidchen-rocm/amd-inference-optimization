"""Firebase ID-token validation using Google's public signing certificates."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import jwt
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa

CERT_URL = (
    "https://www.googleapis.com/robot/v1/metadata/x509/securetoken%40system.gserviceaccount.com"
)


class AuthenticationError(ValueError):
    """An ID token is absent, invalid, or expired. Never include its value."""


class AuthenticationUnavailable(RuntimeError):
    """Signing certificates could not be refreshed."""


@dataclass(frozen=True)
class Identity:
    uid: str


def fetch_certificates() -> tuple[dict[str, str], int]:
    try:
        request = urllib.request.Request(CERT_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read(512 * 1024 + 1)
            if response.status != 200 or len(body) > 512 * 1024:
                raise AuthenticationUnavailable("Sign-in verification is temporarily unavailable.")
            data = json.loads(body)
            match = re.search(r"(?:^|,)\s*max-age=(\d+)", response.headers.get("Cache-Control", ""))
            ttl = min(int(match.group(1)), 86400) if match else 300
            return data, ttl
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise AuthenticationUnavailable(
            "Sign-in verification is temporarily unavailable."
        ) from error


class FirebaseVerifier:
    def __init__(
        self,
        project_id: str,
        *,
        fetch: Callable[[], tuple[dict[str, str], int]] = fetch_certificates,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        clock_skew: int = 30,
    ):
        if not re.fullmatch(r"[a-z][a-z0-9-]{3,100}", project_id):
            raise ValueError("A valid Firebase project ID is required.")
        self.project_id = project_id
        self.issuer = "https://securetoken.google.com/" + project_id
        self.fetch = fetch
        self.clock = clock
        self.monotonic = monotonic
        self.clock_skew = clock_skew
        self._keys: dict[str, rsa.RSAPublicKey] = {}
        self._expires = 0.0
        self._force_after = 0.0
        self._lock = threading.Lock()

    def _key(self, kid: str) -> rsa.RSAPublicKey:
        with self._lock:
            now = self.monotonic()
            expired = now >= self._expires
            if expired or (kid not in self._keys and now >= self._force_after):
                try:
                    certificates, ttl = self.fetch()
                    if not isinstance(certificates, dict) or not 1 <= len(certificates) <= 32:
                        raise ValueError("Invalid certificate response")
                    keys = {}
                    for name, pem in certificates.items():
                        if (
                            not isinstance(name, str)
                            or not isinstance(pem, str)
                            or len(pem) > 16384
                        ):
                            raise ValueError("Invalid signing certificate")
                        key = x509.load_pem_x509_certificate(pem.encode("ascii")).public_key()
                        if not isinstance(key, rsa.RSAPublicKey):
                            raise ValueError("Unsupported signing certificate")
                        keys[name] = key
                    self._keys = keys
                    self._expires = now + max(0, min(int(ttl), 86400))
                    self._force_after = now + 60
                except Exception as error:
                    if isinstance(error, AuthenticationUnavailable):
                        raise
                    raise AuthenticationUnavailable(
                        "Sign-in verification is temporarily unavailable."
                    ) from error
            key = self._keys.get(kid)
            if key is None:
                raise AuthenticationError("Please sign in again.")
            return key

    def verify(self, token: str) -> Identity:
        if not isinstance(token, str) or not token or len(token.encode()) > 8192:
            raise AuthenticationError("Please sign in again.")
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if header.get("alg") != "RS256" or not isinstance(kid, str):
                raise AuthenticationError("Please sign in again.")
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", kid):
                raise AuthenticationError("Please sign in again.")
            claims = jwt.decode(
                token,
                self._key(kid),
                algorithms=["RS256"],
                audience=self.project_id,
                issuer=self.issuer,
                leeway=self.clock_skew,
                options={"require": ["exp", "iat", "aud", "iss", "sub", "auth_time"]},
            )
            now = self.clock()
            if claims["aud"] != self.project_id or claims["iss"] != self.issuer:
                raise AuthenticationError("Please sign in again.")
            for claim in ("iat", "exp", "auth_time"):
                value = claims[claim]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise AuthenticationError("Please sign in again.")
            if (
                claims["exp"] <= now - self.clock_skew
                or claims["iat"] > now + self.clock_skew
                or claims["auth_time"] > now + self.clock_skew
                or claims["auth_time"] > claims["iat"] + self.clock_skew
            ):
                raise AuthenticationError("Please sign in again.")
            uid = claims["sub"]
            if not isinstance(uid, str) or not 1 <= len(uid) <= 128:
                raise AuthenticationError("Please sign in again.")
            if "uid" in claims and claims["uid"] != uid:
                raise AuthenticationError("Please sign in again.")
            return Identity(uid)
        except AuthenticationUnavailable:
            raise
        except (jwt.PyJWTError, ValueError, TypeError, KeyError) as error:
            raise AuthenticationError("Please sign in again.") from error
