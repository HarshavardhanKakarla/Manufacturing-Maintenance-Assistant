import os
import unittest
from io import BytesIO
from unittest.mock import patch

from app import app


class FlaskAppTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page_loads(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Manufacturing Maintenance Assistant", response.data)

    def test_asking_without_index_redirects_with_message(self) -> None:
        response = self.client.post(
            "/ask",
            data={"question": "What does the manual say?"},
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Index the documents before asking", response.data)

    def test_index_requires_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            response = self.client.post("/index", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Set OPENAI_API_KEY", response.data)

    def test_rejects_unsupported_upload_type(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            response = self.client.post(
                "/index",
                data={"documents": (BytesIO(b"example"), "manual.docx")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported file type", response.data)


if __name__ == "__main__":
    unittest.main()
