"""Lexer for the Vibec language with Python-style indentation handling."""

from .tokens import KEYWORDS, Token, TokenType


class LexerError(Exception):
  """Raised when the lexer encounters invalid input."""

  def __init__(self, message: str, line: int, column: int) -> None:
    super().__init__(f"{message} at line {line}, column {column}")
    self.line = line
    self.column = column


class Lexer:
  """Tokenizes Vibec source code, handling indentation-based blocks."""

  def __init__(self, source: str) -> None:
    self.source = source
    self.pos = 0
    self.line = 1
    self.column = 1
    self.indent_stack: list[int] = [0]
    self.tokens: list[Token] = []
    self.at_line_start = True

  def _current(self) -> str:
    if self.pos >= len(self.source):
      return ""
    return self.source[self.pos]

  def _peek(self, offset: int = 1) -> str:
    pos = self.pos + offset
    if pos >= len(self.source):
      return ""
    return self.source[pos]

  def _advance(self) -> str:
    ch = self._current()
    self.pos += 1
    if ch == "\n":
      self.line += 1
      self.column = 1
    else:
      self.column += 1
    return ch

  def _add_token(self, type: TokenType, value: str, line: int, column: int) -> None:
    self.tokens.append(Token(type, value, line, column))

  def _skip_comment(self) -> None:
    while self._current() and self._current() != "\n":
      self._advance()

  def _read_string_while(self, predicate: callable) -> str:
    start = self.pos
    while self._current() and predicate(self._current()):
      self._advance()
    return self.source[start : self.pos]

  def _handle_indentation(self) -> None:
    """Process indentation at the start of a line."""
    line_start = self.line
    col_start = self.column

    # Count spaces (we only support spaces, not tabs)
    indent = 0
    while self._current() == " ":
      self._advance()
      indent += 1

    # Skip blank lines and comments
    if self._current() == "\n" or self._current() == "#" or self._current() == "":
      return

    current_indent = self.indent_stack[-1]

    if indent > current_indent:
      self.indent_stack.append(indent)
      self._add_token(TokenType.INDENT, "", line_start, col_start)
    elif indent < current_indent:
      while self.indent_stack and self.indent_stack[-1] > indent:
        self.indent_stack.pop()
        self._add_token(TokenType.DEDENT, "", line_start, col_start)
      if self.indent_stack[-1] != indent:
        raise LexerError("Inconsistent indentation", line_start, col_start)

  def _read_number(self) -> None:
    line, col = self.line, self.column
    num = self._read_string_while(lambda c: c.isdigit())
    self._add_token(TokenType.INT, num, line, col)

  def _read_identifier(self) -> None:
    line, col = self.line, self.column
    ident = self._read_string_while(lambda c: c.isalnum() or c == "_")
    token_type = KEYWORDS.get(ident, TokenType.IDENT)
    self._add_token(token_type, ident, line, col)

  def tokenize(self) -> list[Token]:
    """Tokenize the entire source and return a list of tokens."""
    while self.pos < len(self.source):
      if self.at_line_start:
        self._handle_indentation()
        self.at_line_start = False

      ch = self._current()
      line, col = self.line, self.column

      if ch == "":
        break
      elif ch == "\n":
        self._advance()
        # Only emit NEWLINE if there's meaningful content before it
        if self.tokens and self.tokens[-1].type not in (
          TokenType.NEWLINE,
          TokenType.INDENT,
        ):
          self._add_token(TokenType.NEWLINE, "\\n", line, col)
        self.at_line_start = True
      elif ch == " ":
        self._advance()
      elif ch == "#":
        self._skip_comment()
      elif ch.isdigit():
        self._read_number()
      elif ch.isalpha() or ch == "_":
        self._read_identifier()
      elif ch == "(":
        self._advance()
        self._add_token(TokenType.LPAREN, "(", line, col)
      elif ch == ")":
        self._advance()
        self._add_token(TokenType.RPAREN, ")", line, col)
      elif ch == ":":
        self._advance()
        self._add_token(TokenType.COLON, ":", line, col)
      elif ch == ",":
        self._advance()
        self._add_token(TokenType.COMMA, ",", line, col)
      elif ch == "+":
        self._advance()
        self._add_token(TokenType.PLUS, "+", line, col)
      elif ch == "-":
        self._advance()
        if self._current() == ">":
          self._advance()
          self._add_token(TokenType.ARROW, "->", line, col)
        else:
          self._add_token(TokenType.MINUS, "-", line, col)
      elif ch == "*":
        self._advance()
        self._add_token(TokenType.STAR, "*", line, col)
      elif ch == "/":
        self._advance()
        self._add_token(TokenType.SLASH, "/", line, col)
      elif ch == "%":
        self._advance()
        self._add_token(TokenType.PERCENT, "%", line, col)
      elif ch == "=":
        self._advance()
        if self._current() == "=":
          self._advance()
          self._add_token(TokenType.EQ, "==", line, col)
        else:
          self._add_token(TokenType.ASSIGN, "=", line, col)
      elif ch == "!":
        self._advance()
        if self._current() == "=":
          self._advance()
          self._add_token(TokenType.NE, "!=", line, col)
        else:
          raise LexerError(f"Unexpected character '!'", line, col)
      elif ch == "<":
        self._advance()
        if self._current() == "=":
          self._advance()
          self._add_token(TokenType.LE, "<=", line, col)
        else:
          self._add_token(TokenType.LT, "<", line, col)
      elif ch == ">":
        self._advance()
        if self._current() == "=":
          self._advance()
          self._add_token(TokenType.GE, ">=", line, col)
        else:
          self._add_token(TokenType.GT, ">", line, col)
      else:
        raise LexerError(f"Unexpected character '{ch}'", line, col)

    # Emit remaining DEDENTs at end of file
    while len(self.indent_stack) > 1:
      self.indent_stack.pop()
      self._add_token(TokenType.DEDENT, "", self.line, self.column)

    self._add_token(TokenType.EOF, "", self.line, self.column)
    return self.tokens


def tokenize(source: str) -> list[Token]:
  """Convenience function to tokenize source code."""
  return Lexer(source).tokenize()
