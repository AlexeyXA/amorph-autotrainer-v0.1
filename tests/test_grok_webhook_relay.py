import base64
import hashlib
import hmac

import pytest

from scripts.relay_grok_webhook import (
    make_payload,
    matches_event,
    sign_payload,
    validate_webhook_url,
)


def sample_event(body="[GPT] Test auto-bridge: please ACK"):
    return {
        "repository": {
            "full_name": "AlexeyXA/amorph-autotrainer-v0.1",
            "id": 1428019829,
        },
        "issue": {
            "number": 1,
            "html_url": "https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1",
        },
        "comment": {
            "id": 12345,
            "html_url": "https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1#issuecomment-12345",
            "body": body,
            "user": {"login": "AlexeyXA"},
        },
    }


def test_matches_only_explicit_owner_gpt_comment():
    assert matches_event(sample_event())
    assert matches_event(sample_event("[GPT] Please ask Grok to review this."))
    assert not matches_event(sample_event("[Grok] ACK"))
    assert not matches_event(sample_event("Please ask Grok to review this."))
    assert not matches_event(sample_event("[GPT] General note with no routing marker."))


def test_matches_event_fails_closed_for_wrong_repo_issue_or_actor():
    event = sample_event()
    event["repository"]["full_name"] = "someone/else"
    assert not matches_event(event)

    event = sample_event()
    event["issue"]["number"] = 2
    assert not matches_event(event)

    event = sample_event()
    event["issue"]["pull_request"] = {"url": "https://api.github.com/repos/example/pulls/1"}
    assert not matches_event(event)

    event = sample_event()
    event["comment"]["user"]["login"] = "unknown-user"
    assert not matches_event(event)


def test_payload_keeps_comment_as_data_and_includes_dedupe_metadata():
    payload = make_payload(sample_event())
    assert payload["event"] == "github.issue_comment.created"
    assert payload["comment_id"] == 12345
    assert payload["comment_body"] == "[GPT] Test auto-bridge: please ACK"
    assert "untrusted content" in payload["safety"]
    assert payload["repository_id"] == 1428019829


def test_standard_webhooks_signature_matches_hmac_sha256():
    key = b"example signing key"
    secret = "whsec_" + base64.b64encode(key).decode("ascii")
    webhook_id = "gh-issue-comment-12345"
    timestamp = "1793181600"
    body = b'{"event":"github.issue_comment.created"}'

    signature = sign_payload(
        secret=secret,
        webhook_id=webhook_id,
        timestamp=timestamp,
        body=body,
    )

    signed_content = f"{webhook_id}.{timestamp}.".encode() + body
    expected = "v1," + base64.b64encode(
        hmac.new(key, signed_content, hashlib.sha256).digest()
    ).decode("ascii")
    assert signature == expected


def test_webhook_url_requires_https_and_no_embedded_credentials():
    validate_webhook_url("https://example.test/webhook")
    with pytest.raises(ValueError):
        validate_webhook_url("http://example.test/webhook")
    with pytest.raises(ValueError):
        validate_webhook_url("https://user:pass@example.test/webhook")
    with pytest.raises(ValueError):
        validate_webhook_url("https://example.test/webhook#fragment")
