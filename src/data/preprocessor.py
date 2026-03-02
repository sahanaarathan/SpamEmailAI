import re
import string
from typing import Optional
from loguru import logger


# Default short-form expansions — can be overridden via config
DEFAULT_SHORT_FORMS: dict[str, str] = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "hv": "have",
    "won": "win",
    "congrats": "congratulations",
    "y": "why",
    "tmrw": "tomorrow",
    "tdy": "today",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "txt": "text",
    "msg": "message",
    "b4": "before",
    "gr8": "great",
    "m8": "mate",
    "luv": "love",
    "asap": "as soon as possible",
    "btw": "by the way",
    "fyi": "for your information",
    "idk": "i do not know",
    "imo": "in my opinion",
    "nvm": "never mind",
    "omg": "oh my god",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "tldr": "too long did not read",
    "brb": "be right back",
}


class TextPreprocessor:
    def __init__(
        self,
        short_forms: Optional[dict] = None,
        max_input_length: int = 5000,
    ):
        self.short_forms = short_forms or DEFAULT_SHORT_FORMS
        self.max_input_length = max_input_length

    def clean(self, text: str) -> str:
        """Full cleaning pipeline: truncate → lowercase → expand → strip → normalize."""
        if not isinstance(text, str):
            text = str(text)

        # Truncate to prevent abuse
        text = text[: self.max_input_length]

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", " url ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " email ", text)

        # Replace phone numbers
        text = re.sub(r"\b\d[\d\s\-\(\)]{7,}\d\b", " phonenumber ", text)

        # Replace currency amounts
        text = re.sub(r"[£$€]\d+[\d,\.]*", " money ", text)

        # Expand short forms (word-boundary aware)
        words = text.split()
        words = [self.short_forms.get(word, word) for word in words]
        text = " ".join(words)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_series(self, series) -> list[str]:
        """Clean an entire pandas Series of messages."""
        logger.info(f"Cleaning {len(series)} messages")
        return [self.clean(msg) for msg in series]
