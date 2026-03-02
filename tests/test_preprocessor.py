import pytest
from src.data.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestTextPreprocessor:
    def test_lowercases_text(self, preprocessor):
        assert preprocessor.clean("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self, preprocessor):
        result = preprocessor.clean("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_expands_short_forms(self, preprocessor):
        result = preprocessor.clean("u r my friend")
        assert "you" in result
        assert "are" in result

    def test_removes_url(self, preprocessor):
        result = preprocessor.clean("Visit http://example.com for details")
        assert "http" not in result
        assert "url" in result

    def test_removes_email(self, preprocessor):
        result = preprocessor.clean("Email us at support@company.com")
        assert "@" not in result
        assert "email" in result

    def test_replaces_phone_number(self, preprocessor):
        result = preprocessor.clean("Call us at 08001234567 now!")
        assert "phonenumber" in result

    def test_replaces_currency(self, preprocessor):
        result = preprocessor.clean("Win £500 cash prize")
        assert "money" in result

    def test_truncates_long_input(self, preprocessor):
        long_text = "a" * 10000
        result = preprocessor.clean(long_text)
        assert len(result) <= 5000

    def test_handles_empty_string(self, preprocessor):
        result = preprocessor.clean("")
        assert result == ""

    def test_handles_non_string_input(self, preprocessor):
        result = preprocessor.clean(12345)
        assert isinstance(result, str)

    def test_collapses_whitespace(self, preprocessor):
        result = preprocessor.clean("hello    world")
        assert "  " not in result

    def test_clean_series(self, preprocessor):
        import pandas as pd
        series = pd.Series(["Hello World", "U win a prize"])
        results = preprocessor.clean_series(series)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_custom_short_forms(self):
        custom = TextPreprocessor(short_forms={"ty": "thank you"})
        result = custom.clean("ty for the gift")
        assert "thank you" in result

    def test_spam_message_cleaned(self, preprocessor):
        spam = "Congratulations! U hv won £500. Call 08001234567 or visit http://claim.com"
        result = preprocessor.clean(spam)
        assert "£" not in result
        assert "http" not in result
        assert "you" in result or "have" in result
