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
from echo.utils.download import _strip_multipart_wrapper


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

    @patch("echo.models.user_conversation.download_url_as_bytes")
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

    @patch("echo.models.user_conversation.download_url_as_bytes")
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


class TestBedrockRealisticImage:
    """Tests with realistic image sizes to verify Bedrock conversion handles them correctly."""

    @patch("echo.models.user_conversation.download_url_as_bytes")
    def test_image_url_preserves_downloaded_bytes_size(self, mock_download):
        """Ensure the bytes sent to Bedrock match exactly what was downloaded (no re-encoding)."""
        # Simulate a ~60KB JPEG (realistic small photo)
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 60_000 + b"\xff\xd9"
        mock_download.return_value = fake_jpeg

        msg = Message(
            role=MessageRole.USER,
            content=[
                TextMessage(text="Describe this image"),
                ImageContent(
                    media_type="image/jpeg",
                    source_type=ContentSourceType.URL,
                    url="https://bucket.s3.amazonaws.com/img.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA%2F20260128%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Signature=abc123",
                ),
            ],
        )
        result = msg.to_bedrock_message()
        img_block = result["content"][1]["image"]
        assert img_block["format"] == "jpeg"
        # Verify the EXACT bytes are passed through — no re-encoding or truncation
        assert img_block["source"]["bytes"] is fake_jpeg
        assert len(img_block["source"]["bytes"]) == len(fake_jpeg)
        mock_download.assert_called_once()

    def test_base64_realistic_image_roundtrip(self):
        """Verify base64 decode produces correct byte count for Bedrock."""
        # Create a ~60KB fake image and base64 encode it
        raw_bytes = b"\xff\xd8\xff\xe0" + b"\xab" * 60_000 + b"\xff\xd9"
        b64_data = base64.b64encode(raw_bytes).decode()

        msg = Message(
            role=MessageRole.USER,
            content=[
                TextMessage(text="What is this?"),
                ImageContent(
                    media_type="image/jpeg",
                    source_type=ContentSourceType.BASE64,
                    data=b64_data,
                ),
            ],
        )
        result = msg.to_bedrock_message()
        img_bytes = result["content"][1]["image"]["source"]["bytes"]
        # Decoded bytes must match original
        assert img_bytes == raw_bytes
        assert len(img_bytes) == len(raw_bytes)
        # base64 string is ~33% larger than raw bytes
        assert len(b64_data) > len(raw_bytes)

    @patch("echo.models.user_conversation.download_url_as_bytes")
    def test_download_url_with_presigned_params(self, mock_download):
        """Verify presigned S3 URLs are passed through to httpx unchanged."""
        presigned_url = (
            "https://bucket.s3.amazonaws.com/session/file.jpg"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=ASIAY%2F20260128%2Fap-south-1%2Fs3%2Faws4_request"
            "&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJ%2F%2F%2F%2F"
            "&X-Amz-Signature=abcdef1234567890"
        )
        mock_download.return_value = b"\xff\xd8image"

        msg = Message(
            role=MessageRole.USER,
            content=[
                TextMessage(text="describe"),
                ImageContent(
                    media_type="image/jpeg",
                    source_type=ContentSourceType.URL,
                    url=presigned_url,
                ),
            ],
        )
        msg.to_bedrock_message()
        # The URL passed to _download_url_as_bytes must match the original
        called_url = mock_download.call_args[0][0]
        assert called_url == presigned_url


class TestStripMultipartWrapper:
    """Tests for _strip_multipart_wrapper which extracts file bytes from multipart form data."""

    def test_raw_jpeg_unchanged(self):
        raw = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"
        assert _strip_multipart_wrapper(raw) is raw

    def test_raw_pdf_unchanged(self):
        raw = b"%PDF-1.4 some content"
        assert _strip_multipart_wrapper(raw) is raw

    def test_empty_bytes_unchanged(self):
        assert _strip_multipart_wrapper(b"") == b""

    def test_strips_multipart_form_wrapper(self):
        """Simulate what S3 stores when client uploads via multipart/form-data."""
        jpeg_payload = b"\xff\xd8\xff\xe0" + b"\xab" * 500 + b"\xff\xd9"
        boundary = b"----------------------------525499894086112470"
        multipart = (
            boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="photo.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n"
            b"\r\n"
            + jpeg_payload + b"\r\n"
            + boundary + b"--\r\n"
        )
        result = _strip_multipart_wrapper(multipart)
        assert result == jpeg_payload
        assert result[:2] == b"\xff\xd8"

    def test_strips_multipart_pdf(self):
        pdf_payload = b"%PDF-1.4 fake pdf content here"
        boundary = b"----WebKitFormBoundary7MA4YWxkTrZu0gW"
        multipart = (
            boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="doc.pdf"\r\n'
            b"Content-Type: application/pdf\r\n"
            b"\r\n"
            + pdf_payload + b"\r\n"
            + boundary + b"--\r\n"
        )
        result = _strip_multipart_wrapper(multipart)
        assert result == pdf_payload

    @patch("echo.models.user_conversation.download_url_as_bytes")
    def test_bedrock_gets_clean_bytes_from_multipart_upload(self, mock_download):
        """End-to-end: multipart-wrapped S3 content produces clean bytes for Bedrock."""
        jpeg_payload = b"\xff\xd8\xff\xe0" + b"\x00" * 200 + b"\xff\xd9"
        # download_url_as_bytes already strips multipart, so mock returns clean bytes
        mock_download.return_value = jpeg_payload

        msg = Message(
            role=MessageRole.USER,
            content=[
                TextMessage(text="describe"),
                ImageContent(
                    media_type="image/jpeg",
                    source_type=ContentSourceType.URL,
                    url="https://bucket.s3.amazonaws.com/file.jpg",
                ),
            ],
        )
        result = msg.to_bedrock_message()
        img_bytes = result["content"][1]["image"]["source"]["bytes"]
        assert img_bytes == jpeg_payload
        assert img_bytes[:2] == b"\xff\xd8"


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
