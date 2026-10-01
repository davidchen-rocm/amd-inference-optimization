from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from macfit_training.service.auth import (
    AuthenticationError,
    AuthenticationUnavailable,
    FirebaseVerifier,
)

PROJECT = "macfit-example"


@pytest.fixture
def signer():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Synthetic test signer")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM).decode()

    def token(**changes):
        claims = {
            "sub": "alice",
            "aud": PROJECT,
            "iss": "https://securetoken.google.com/" + PROJECT,
            "iat": int(time.time()) - 2,
            "exp": int(time.time()) + 3600,
            "auth_time": int(time.time()) - 5,
        }
        claims.update(changes)
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})

    return pem, token


def test_firebase_identity_uses_verified_subject_and_cached_public_certificate(signer):
    pem, token = signer
    calls = []
    verifier = FirebaseVerifier(PROJECT, fetch=lambda: (calls.append(1) or {"test-key": pem}, 600))
    assert verifier.verify(token()).uid == "alice"
    assert verifier.verify(token()).uid == "alice"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"aud": "another-project"},
        {"iss": "https://accounts.google.com"},
        {"sub": ""},
        {"uid": "forged-owner"},
        {"exp": 1},
        {"auth_time": 9999999999},
        {"iat": 9999999999},
        {"auth_time": "123"},
    ],
)
def test_firebase_rejects_wrong_project_issuer_owner_and_time(signer, changes):
    pem, token = signer
    verifier = FirebaseVerifier(PROJECT, fetch=lambda: ({"test-key": pem}, 600))
    with pytest.raises(AuthenticationError):
        verifier.verify(token(**changes))


def test_algorithm_confusion_is_rejected_before_fetching_a_certificate():
    verifier = FirebaseVerifier(PROJECT, fetch=lambda: pytest.fail("must not fetch an HS256 key"))
    token = jwt.encode(
        {"sub": "alice"},
        "synthetic-long-test-secret-value-only",
        algorithm="HS256",
        headers={"kid": "test-key"},
    )
    with pytest.raises(AuthenticationError):
        verifier.verify(token)


def test_expired_certificate_cache_cannot_be_used_when_refresh_fails(signer):
    pem, token = signer
    clock, calls = [1.0], []

    def fetch():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("offline")
        return {"test-key": pem}, 10

    verifier = FirebaseVerifier(PROJECT, fetch=fetch, monotonic=lambda: clock[0])
    assert verifier.verify(token()).uid == "alice"
    clock[0] = 12
    with pytest.raises(AuthenticationUnavailable):
        verifier.verify(token())


def test_unknown_key_ids_do_not_trigger_unbounded_certificate_refreshes(signer):
    pem, token = signer
    calls = []
    verifier = FirebaseVerifier(PROJECT, fetch=lambda: (calls.append(1) or {"other-key": pem}, 600))
    for _ in range(10):
        with pytest.raises(AuthenticationError):
            verifier.verify(token())
    assert len(calls) == 1


def test_audience_must_be_the_single_configured_firebase_project(signer):
    pem, token = signer
    verifier = FirebaseVerifier(PROJECT, fetch=lambda: ({"test-key": pem}, 3600))
    with pytest.raises(AuthenticationError):
        verifier.verify(token(aud=[PROJECT, "another-project"]))
