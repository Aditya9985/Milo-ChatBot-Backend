import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

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
            main.client.chat.completions, "create", return_value=completion
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

    def test_chat_rejects_empty_and_oversized_messages(self) -> None:
        empty_response = self.client.post("/chat", json={"message": "   "})
        long_response = self.client.post("/chat", json={"message": "x" * 4001})

        self.assertEqual(empty_response.status_code, 422)
        self.assertEqual(long_response.status_code, 422)

    def test_tts_returns_audio(self) -> None:
        speech = SimpleNamespace(content=b"audio data")
        with patch.object(
            main.client.audio.speech, "create", return_value=speech
        ) as create:
            response = self.client.post("/tts", json={"text": "Hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.content, b"audio data")
        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
