import unittest
from unittest.mock import AsyncMock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

import main


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        main.app.dependency_overrides[main.require_firebase_user] = lambda: {
            "sub": "user_test"
        }

    def tearDown(self) -> None:
        main.app.dependency_overrides.clear()

    def test_health_check(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_chat_is_stateless_and_returns_current_turn(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello! How can I help?")
                )
            ]
        )
        with patch.object(
            main.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=completion,
        ) as create:
            response = self.client.post("/chat", json={"message": "  Hi!  "})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Hello! How can I help?")
        self.assertEqual(
            [message["role"] for message in response.json()["history"]],
            ["user", "assistant"],
        )
        self.assertEqual(response.json()["history"][0]["content"], "Hi!")
        self.assertEqual(len(create.call_args.kwargs["messages"]), 2)
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 500)

    def test_suzu_chat_uses_suzu_persona(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello! I’m Suzu.")
                )
            ]
        )
        with patch.object(
            main.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=completion,
        ) as create:
            response = self.client.post(
                "/chat",
                json={"message": "Hello", "companion": "suzu"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            create.call_args.kwargs["messages"][0]["content"],
            main.SUZU_SYSTEM_PROMPT,
        )

    def test_voice_chat_uses_brief_response_instructions(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello! What can I help with?")
                )
            ]
        )
        with patch.object(
            main.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=completion,
        ) as create:
            response = self.client.post(
                "/chat",
                json={"message": "Hello", "voice_mode": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "one brief sentence of at most 18 words",
            create.call_args.kwargs["messages"][0]["content"],
        )
        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 120)

    def test_chat_rejects_empty_and_oversized_messages(self) -> None:
        empty_response = self.client.post("/chat", json={"message": "   "})
        long_response = self.client.post("/chat", json={"message": "x" * 4001})

        self.assertEqual(empty_response.status_code, 422)
        self.assertEqual(long_response.status_code, 422)

    def test_chat_requires_authentication(self) -> None:
        main.app.dependency_overrides.clear()

        response = self.client.post("/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_authenticated_request_requires_firebase_project_configuration(self) -> None:
        main.app.dependency_overrides.clear()
        with patch.object(main, "FIREBASE_PROJECT_ID", ""):
            response = self.client.post(
                "/chat",
                json={"message": "Hello"},
                headers={"Authorization": "Bearer test-token"},
            )

        self.assertEqual(response.status_code, 503)

    def test_firebase_id_token_verification_checks_issuer_audience_and_signature(
        self,
    ) -> None:
        signing_key = SimpleNamespace(key="public-key")
        jwks_client = SimpleNamespace(
            get_signing_key_from_jwt=Mock(return_value=signing_key)
        )
        project_id = "milo-test-project"
        issuer = f"https://securetoken.google.com/{project_id}"
        with (
            patch.object(main, "FIREBASE_PROJECT_ID", project_id),
            patch.object(main, "FIREBASE_ISSUER", issuer),
            patch.object(main, "FIREBASE_JWKS_CLIENT", jwks_client),
            patch.object(
                main.jwt, "decode", return_value={"sub": "user_123"}
            ) as decode,
        ):
            claims = main.require_firebase_user(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
            )

        self.assertEqual(claims["sub"], "user_123")
        decode.assert_called_once_with(
            "token",
            "public-key",
            algorithms=["RS256"],
            audience=project_id,
            issuer=issuer,
            options={"require": ["exp", "iat", "auth_time", "iss", "sub"]},
        )

    def test_invalid_firebase_id_token_is_rejected(self) -> None:
        project_id = "milo-test-project"
        jwks_client = SimpleNamespace(
            get_signing_key_from_jwt=Mock(
                return_value=SimpleNamespace(key="public-key")
            )
        )
        with (
            patch.object(main, "FIREBASE_PROJECT_ID", project_id),
            patch.object(
                main, "FIREBASE_ISSUER", f"https://securetoken.google.com/{project_id}"
            ),
            patch.object(main, "FIREBASE_JWKS_CLIENT", jwks_client),
            patch.object(main.jwt, "decode", side_effect=main.InvalidTokenError),
        ):
            with self.assertRaises(main.HTTPException) as error:
                main.require_firebase_user(
                    HTTPAuthorizationCredentials(
                        scheme="Bearer", credentials="invalid-token"
                    )
                )

        self.assertEqual(error.exception.status_code, 401)

    def test_tts_route_is_removed(self) -> None:
        response = self.client.post("/tts", json={"text": "Hello"})

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
