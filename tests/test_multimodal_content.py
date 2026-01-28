"""Unit tests for ImageContent and DocumentContent support."""

import base64
from unittest.mock import patch, MagicMock

import pytest

from echo.models.user_conversation import (
    ContentSourceType,
    DocumentContent,
    ImageContent,
    Message,
    MessageRole,
    TextMessage,
)


# --------- Validation Tests ---------


class TestImageContentValidation:
    def test_base64_image_valid(self):
        img = ImageContent(
            media_type="image/png",
            source_type=ContentSourceType.BASE64,
            data="iVBORw0KGgo=",
        )
        assert img.data == "iVBORw0KGgo="

    def test_url_image_valid(self):
        img = ImageContent(
            media_type="image/jpeg",
            source_type=ContentSourceType.URL,
            url="https://example.com/img.jpg",
        )
        assert str(img.url) == "https://example.com/img.jpg"

    def test_base64_image_missing_data(self):
        with pytest.raises(ValueError, match="data is required"):
            ImageContent(
                media_type="image/png",
                source_type=ContentSourceType.BASE64,
            )

    def test_url_image_missing_url(self):
        with pytest.raises(ValueError, match="url is required"):
            ImageContent(
                media_type="image/png",
                source_type=ContentSourceType.URL,
            )


class TestDocumentContentValidation:
    def test_base64_doc_valid(self):
        doc = DocumentContent(
            media_type="application/pdf",
            source_type=ContentSourceType.BASE64,
            data="JVBERi0=",
            name="report.pdf",
        )
        assert doc.name == "report.pdf"

    def test_url_doc_valid(self):
        doc = DocumentContent(
            media_type="application/pdf",
            source_type=ContentSourceType.URL,
            url="https://example.com/doc.pdf",
        )
        assert str(doc.url) == "https://example.com/doc.pdf"

    def test_base64_doc_missing_data(self):
        with pytest.raises(ValueError, match="data is required"):
            DocumentContent(
                media_type="application/pdf",
                source_type=ContentSourceType.BASE64,
            )

    def test_url_doc_missing_url(self):
        with pytest.raises(ValueError, match="url is required"):
            DocumentContent(
                media_type="application/pdf",
                source_type=ContentSourceType.URL,
            )


class TestMessageValidation:
    def test_user_message_with_image(self):
        msg = Message(
            role=MessageRole.USER,
            content=[
                TextMessage(text="What is in this image?"),
                ImageContent(
                    media_type="image/png",
                    source_type=ContentSourceType.BASE64,
                    data="iVBORw0KGgo=",
                ),
            ],
        )
        assert len(msg.content) == 2

    def test_user_message_with_document(self):
        msg = Message(
            role=MessageRole.USER,
            content=[
                TextMessage(text="Summarize this PDF"),
                DocumentContent(
                    media_type="application/pdf",
                    source_type=ContentSourceType.BASE64,
                    data="JVBERi0=",
                ),
            ],
        )
        assert len(msg.content) == 2


# --------- Provider Conversion Tests ---------

B64_PNG = base64.b64encode(b"\x89PNG").decode()
B64_PDF = base64.b64encode(b"%PDF-").decode()


def _user_msg_with_image(source_type, **kwargs):
    return Message(
        role=MessageRole.USER,
        content=[
            TextMessage(text="Describe this"),
            ImageContent(media_type="image/png", source_type=source_type, **kwargs),
        ],
    )


def _user_msg_with_doc(source_type, **kwargs):
    return Message(
        role=MessageRole.USER,
        content=[
            TextMessage(text="Summarize"),
            DocumentContent(media_type="application/pdf", source_type=source_type, name="test.pdf", **kwargs),
        ],
    )


class TestAnthropicConversion:
    def test_image_base64(self):
        result = _user_msg_with_image(ContentSourceType.BASE64, data=B64_PNG).to_anthropic_message()
        assert result["role"] == "user"
        img_block = result["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["data"] == B64_PNG

    def test_image_url(self):
        result = _user_msg_with_image(ContentSourceType.URL, url="https://example.com/i.png").to_anthropic_message()
        img_block = result["content"][1]
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/i.png"

    def test_document_base64(self):
        result = _user_msg_with_doc(ContentSourceType.BASE64, data=B64_PDF).to_anthropic_message()
        doc_block = result["content"][1]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "base64"


class TestOpenAIConversion:
    def test_image_base64(self):
        msgs = _user_msg_with_image(ContentSourceType.BASE64, data=B64_PNG).to_openai_messages()
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_url(self):
        msgs = _user_msg_with_image(ContentSourceType.URL, url="https://example.com/i.png").to_openai_messages()
        content = msgs[0]["content"]
        assert content[1]["image_url"]["url"] == "https://example.com/i.png"


class TestBedrockConversion:
    def test_image_base64(self):
        result = _user_msg_with_image(ContentSourceType.BASE64, data=B64_PNG).to_bedrock_message()
        img_block = result["content"][1]
        assert "image" in img_block
        assert img_block["image"]["format"] == "png"
        assert img_block["image"]["source"]["bytes"] == base64.b64decode(B64_PNG)

    @patch("echo.models.user_conversation._download_url_as_bytes")
    def test_image_url(self, mock_download):
        fake_bytes = b"\x89PNG fake image data"
        mock_download.return_value = fake_bytes
        result = _user_msg_with_image(ContentSourceType.URL, url="https://example.com/i.png").to_bedrock_message()
        img_block = result["content"][1]
        assert "image" in img_block
        assert img_block["image"]["format"] == "png"
        assert img_block["image"]["source"]["bytes"] == fake_bytes
        mock_download.assert_called_once_with("https://example.com/i.png")

    def test_document_base64(self):
        result = _user_msg_with_doc(ContentSourceType.BASE64, data=B64_PDF).to_bedrock_message()
        doc_block = result["content"][1]
        assert "document" in doc_block
        assert doc_block["document"]["name"] == "test.pdf"

    @patch("echo.models.user_conversation._download_url_as_bytes")
    def test_document_url(self, mock_download):
        fake_bytes = b"%PDF- fake pdf data"
        mock_download.return_value = fake_bytes
        result = _user_msg_with_doc(ContentSourceType.URL, url="https://example.com/doc.pdf").to_bedrock_message()
        doc_block = result["content"][1]
        assert "document" in doc_block
        assert doc_block["document"]["format"] == "pdf"
        assert doc_block["document"]["source"]["bytes"] == fake_bytes
        assert doc_block["document"]["name"] == "test.pdf"
        mock_download.assert_called_once_with("https://example.com/doc.pdf")


class TestGeminiConversion:
    def test_image_base64(self):
        result = _user_msg_with_image(ContentSourceType.BASE64, data=B64_PNG).to_gemini_message()
        img_part = result["parts"][1]
        assert img_part["inline_data"]["mime_type"] == "image/png"
        assert img_part["inline_data"]["data"] == B64_PNG

    def test_image_url(self):
        result = _user_msg_with_image(ContentSourceType.URL, url="https://example.com/i.png").to_gemini_message()
        img_part = result["parts"][1]
        assert img_part["file_data"]["file_uri"] == "https://example.com/i.png"

    def test_document_base64(self):
        result = _user_msg_with_doc(ContentSourceType.BASE64, data=B64_PDF).to_gemini_message()
        doc_part = result["parts"][1]
        assert doc_part["inline_data"]["mime_type"] == "application/pdf"
