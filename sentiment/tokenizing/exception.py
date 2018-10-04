class TokenizingException(Exception):
    pass


class TokenInitializationException(TokenizingException):
    pass


class EmptyTextValueException(TokenInitializationException):
    pass


class RedundantValueException(TokenInitializationException):
    pass


class WrongArgumentException(TokenizingException):
    pass
