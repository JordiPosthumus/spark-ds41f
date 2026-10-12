"""Only these exact multiword phrases and explicit case variants are added."""
BASE_PHRASES=('Let me go','Let me write the command')
PHRASES=tuple(form for phrase in BASE_PHRASES for form in (phrase,phrase.lower(),phrase.upper()))
