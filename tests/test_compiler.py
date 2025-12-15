"""Tests for the Vibec compiler."""

import tempfile
import subprocess
from pathlib import Path

import pytest

from vibec.lexer import tokenize
from vibec.parser import parse
from vibec.tokens import TokenType
from vibec.checker import check
from vibec.codegen import generate


class TestLexer:
  def test_basic_tokens(self):
    source = "fn main() -> i64:"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert types == [
      TokenType.FN,
      TokenType.IDENT,
      TokenType.LPAREN,
      TokenType.RPAREN,
      TokenType.ARROW,
      TokenType.IDENT,
      TokenType.COLON,
      TokenType.EOF,
    ]

  def test_indentation(self):
    source = """fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.INDENT in types
    assert TokenType.DEDENT in types

  def test_operators(self):
    source = "1 + 2 * 3 - 4 / 5 % 6"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.PLUS in types
    assert TokenType.STAR in types
    assert TokenType.MINUS in types
    assert TokenType.SLASH in types
    assert TokenType.PERCENT in types

  def test_comparisons(self):
    source = "a == b != c < d > e <= f >= g"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.EQ in types
    assert TokenType.NE in types
    assert TokenType.LT in types
    assert TokenType.GT in types
    assert TokenType.LE in types
    assert TokenType.GE in types

  def test_keywords(self):
    source = "fn let struct impl enum match self if else while for in range return and or not true false"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.FN in types
    assert TokenType.LET in types
    assert TokenType.STRUCT in types
    assert TokenType.IMPL in types
    assert TokenType.ENUM in types
    assert TokenType.MATCH in types
    assert TokenType.SELF in types
    assert TokenType.IF in types
    assert TokenType.ELSE in types
    assert TokenType.WHILE in types
    assert TokenType.FOR in types
    assert TokenType.IN in types
    assert TokenType.RANGE in types
    assert TokenType.RETURN in types
    assert TokenType.AND in types
    assert TokenType.OR in types
    assert TokenType.NOT in types
    assert TokenType.TRUE in types
    assert TokenType.FALSE in types

  def test_coloncolon(self):
    source = "Option::Some"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.IDENT in types
    assert TokenType.COLONCOLON in types

  def test_braces(self):
    source = "{ }"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.LBRACE in types
    assert TokenType.RBRACE in types

  def test_string_literal(self):
    source = '"hello world"'
    tokens = tokenize(source)
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "hello world"

  def test_string_escape_sequences(self):
    source = r'"line1\nline2\ttab\\"'
    tokens = tokenize(source)
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "line1\nline2\ttab\\"

  def test_comment_line(self):
    source = """# this is a comment
fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    # Comment should be skipped, first token should be FN
    assert tokens[0].type == TokenType.FN

  def test_comment_after_code(self):
    source = """fn main() -> i64:
    return 42  # inline comment
"""
    tokens = tokenize(source)
    # Should parse correctly, comment ignored
    types = [t.type for t in tokens]
    assert TokenType.RETURN in types
    assert TokenType.INT in types

  def test_comment_only_lines(self):
    source = """fn main() -> i64:
    # comment line 1
    # comment line 2
    return 0
"""
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.RETURN in types


class TestParser:
  def test_simple_function(self):
    source = """fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "main"
    assert ast.functions[0].return_type.name == "i64"

  def test_function_with_params(self):
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    func = ast.functions[0]
    assert func.name == "add"
    assert len(func.params) == 2
    assert func.params[0].name == "a"
    assert func.params[1].name == "b"

  def test_let_statement(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt

    assert isinstance(ast.functions[0].body[0], LetStmt)

  def test_if_statement(self):
    source = """fn main() -> i64:
    if true:
        return 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import IfStmt

    assert isinstance(ast.functions[0].body[0], IfStmt)

  def test_while_statement(self):
    source = """fn main() -> i64:
    while false:
        return 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import WhileStmt

    assert isinstance(ast.functions[0].body[0], WhileStmt)

  def test_string_literal(self):
    source = """fn main() -> i64:
    print("hello")
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import CallExpr, ExprStmt, StringLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, CallExpr)
    assert isinstance(stmt.expr.args[0], StringLiteral)
    assert stmt.expr.args[0].value == "hello"

  def test_assignment(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = 2
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import AssignStmt

    assert isinstance(ast.functions[0].body[1], AssignStmt)
    assert ast.functions[0].body[1].name == "x"

  def test_for_loop(self):
    source = """fn main() -> i64:
    for i in range(5):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ForStmt

    assert isinstance(ast.functions[0].body[0], ForStmt)
    assert ast.functions[0].body[0].var == "i"

  def test_for_loop_with_start(self):
    source = """fn main() -> i64:
    for i in range(2, 5):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ForStmt, IntLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, ForStmt)
    assert isinstance(stmt.start, IntLiteral)
    assert stmt.start.value == 2

  def test_array_type(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [1, 2, 3]
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, ArrayType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, ArrayType)
    assert stmt.type_ann.size == 3

  def test_vec_type(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, VecType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, VecType)

  def test_index_expr(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [1, 2, 3]
    return arr[0]
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import IndexExpr, ReturnStmt

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, IndexExpr)

  def test_method_call(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(1)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ExprStmt, MethodCallExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, MethodCallExpr)
    assert stmt.expr.method == "push"

  def test_struct_definition(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import StructDef

    assert len(ast.structs) == 1
    struct = ast.structs[0]
    assert isinstance(struct, StructDef)
    assert struct.name == "Point"
    assert len(struct.fields) == 2
    assert struct.fields[0].name == "x"
    assert struct.fields[1].name == "y"

  def test_struct_literal(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, StructLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, StructLiteral)
    assert stmt.value.name == "Point"
    assert len(stmt.value.fields) == 2

  def test_field_access(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ReturnStmt, FieldAccessExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, FieldAccessExpr)

  def test_tuple_type(self):
    source = """fn main() -> i64:
    let t: (i64, bool) = (10, true)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, TupleType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, TupleType)
    assert len(stmt.type_ann.element_types) == 2

  def test_tuple_literal(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, TupleLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, TupleLiteral)
    assert len(stmt.value.elements) == 2

  def test_tuple_index(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return t.0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ReturnStmt, TupleIndexExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, TupleIndexExpr)
    assert stmt.value.index == 0

  def test_enum_definition(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import EnumDef

    assert len(ast.enums) == 1
    enum = ast.enums[0]
    assert isinstance(enum, EnumDef)
    assert enum.name == "Option"
    assert len(enum.variants) == 2
    assert enum.variants[0].name == "Some"
    assert enum.variants[0].payload_type is not None
    assert enum.variants[1].name == "None"
    assert enum.variants[1].payload_type is None

  def test_enum_literal(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, EnumLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, EnumLiteral)
    assert stmt.value.enum_name == "Option"
    assert stmt.value.variant_name == "Some"

  def test_match_expression(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ExprStmt, MatchExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, MatchExpr)
    assert len(stmt.expr.arms) == 2

  def test_impl_block(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ImplBlock

    assert len(ast.impls) == 1
    impl = ast.impls[0]
    assert isinstance(impl, ImplBlock)
    assert impl.struct_name == "Point"
    assert len(impl.methods) == 1
    assert impl.methods[0].name == "sum"

  def test_impl_self_parameter(self):
    source = """struct Point:
    x: i64

impl Point:
    fn get_x(self) -> i64:
        return self.x

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import Parameter, SimpleType

    method = ast.impls[0].methods[0]
    assert len(method.params) == 1
    assert isinstance(method.params[0], Parameter)
    assert method.params[0].name == "self"
    assert isinstance(method.params[0].type_ann, SimpleType)
    assert method.params[0].type_ann.name == "Self"


class TestChecker:
  def test_type_mismatch(self):
    source = """fn main() -> i64:
    let x: bool = 42
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_undefined_variable(self):
    source = """fn main() -> i64:
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_undefined_function(self):
    source = """fn main() -> i64:
    return foo()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_wrong_return_type(self):
    source = """fn main() -> i64:
    return true
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_valid_program(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_string_type(self):
    source = """fn main() -> i64:
    let s: str = "hello"
    print(s)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_string_type_mismatch(self):
    source = """fn main() -> i64:
    let s: str = 42
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_assignment_valid(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = 2
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_assignment_type_mismatch(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = true
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_assignment_undefined_variable(self):
    source = """fn main() -> i64:
    x = 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_for_loop_valid(self):
    source = """fn main() -> i64:
    for i in range(10):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_for_loop_invalid_start(self):
    source = """fn main() -> i64:
    for i in range(true, 10):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_valid(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_struct_missing_field(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_unknown_field(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20, z: 30 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_field_type_mismatch(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: true, y: 20 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_undefined(self):
    source = """fn main() -> i64:
    let p: Unknown = Unknown { x: 10 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_tuple_valid(self):
    source = """fn main() -> i64:
    let t: (i64, bool) = (42, true)
    return t.0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_tuple_index_out_of_bounds(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return t.5
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_tuple_type_mismatch(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, true)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_valid(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_enum_unknown(self):
    source = """fn main() -> i64:
    let x: Unknown = Unknown::Variant(1)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_unknown_variant(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Unknown(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_payload_mismatch(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(true)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_missing_payload(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_unexpected_payload(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::None(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_match_exhaustive(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_match_non_exhaustive(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError, match="Non-exhaustive match"):
      check(ast)

  def test_impl_valid(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.sum()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_impl_unknown_struct(self):
    source = """impl Unknown:
    fn foo(self) -> i64:
        return 0

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError, match="unknown type 'Unknown'"):
      check(ast)

  def test_impl_method_not_found(self):
    source = """struct Point:
    x: i64

fn main() -> i64:
    let p: Point = Point { x: 10 }
    return p.nonexistent()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)


class TestCodegen:
  def test_generates_assembly(self):
    source = """fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)
    asm = generate(ast)
    assert ".globl _main" in asm
    assert "_main:" in asm
    assert "ret" in asm


@pytest.mark.skipif(
  subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip() != "arm64",
  reason="ARM64 binary execution tests only run on ARM64 macOS",
)
class TestEndToEnd:
  """End-to-end tests that compile and run actual binaries."""

  def _compile_and_run(self, source: str) -> tuple[int, str]:
    """Compile source and run the binary, returning (exit_code, stdout)."""
    from vibec.compiler import Compiler

    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = Path(tmpdir) / "test_binary"
      compiler = Compiler()
      result = compiler.compile_to_binary(source, output_path)
      assert result.success, f"Compilation failed: {result.error}"

      proc = subprocess.run([str(output_path)], capture_output=True, text=True)
      return proc.returncode, proc.stdout

  def test_return_value(self):
    source = """fn main() -> i64:
    return 42
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_arithmetic(self):
    source = """fn main() -> i64:
    return 2 + 3 * 4
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 14  # 2 + (3 * 4) = 14

  def test_arithmetic_subtraction(self):
    source = """fn main() -> i64:
    return 10 - 3
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_arithmetic_division(self):
    source = """fn main() -> i64:
    return 20 / 4
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5

  def test_arithmetic_modulo(self):
    source = """fn main() -> i64:
    return 17 % 5
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 2

  def test_arithmetic_precedence(self):
    source = """fn main() -> i64:
    return 2 + 3 * 4 - 8 / 2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10  # 2 + 12 - 4 = 10

  def test_arithmetic_unary_minus(self):
    source = """fn main() -> i64:
    let x: i64 = 5
    return -x + 10
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5

  def test_print(self):
    source = """fn main() -> i64:
    print(42)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "42"

  def test_factorial(self):
    source = """fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main() -> i64:
    print(factorial(5))
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "120"

  def test_fibonacci(self):
    source = """fn fib(n: i64) -> i64:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fn main() -> i64:
    print(fib(10))
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "55"

  def test_print_string(self):
    source = """fn main() -> i64:
    print("Hello, World!")
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "Hello, World!"

  def test_string_variable(self):
    source = """fn main() -> i64:
    let msg: str = "Vibec"
    print(msg)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "Vibec"

  def test_string_escape_sequences(self):
    source = r"""fn main() -> i64:
    print("line1\nline2")
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "line1\nline2"

  def test_variable_reassignment(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = 2
    x = 3
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_reassignment_with_expression(self):
    source = """fn main() -> i64:
    let x: i64 = 10
    x = x + 5
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_counter_loop(self):
    source = """fn main() -> i64:
    let count: i64 = 0
    let i: i64 = 0
    while i < 5:
        count = count + 1
        i = i + 1
    return count
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5

  def test_for_loop_simple(self):
    source = """fn main() -> i64:
    let sum: i64 = 0
    for i in range(5):
        sum = sum + i
    return sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10  # 0 + 1 + 2 + 3 + 4 = 10

  def test_for_loop_with_start(self):
    source = """fn main() -> i64:
    let sum: i64 = 0
    for i in range(2, 5):
        sum = sum + i
    return sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 9  # 2 + 3 + 4 = 9

  def test_for_loop_print(self):
    source = """fn main() -> i64:
    for i in range(3):
        print(i)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "0\n1\n2"

  def test_for_loop_nested(self):
    source = """fn main() -> i64:
    let count: i64 = 0
    for i in range(3):
        for j in range(4):
            count = count + 1
    return count
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 12  # 3 * 4 = 12

  def test_array_literal_and_access(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [10, 20, 30]
    return arr[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  def test_array_assignment(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [1, 2, 3]
    arr[0] = 100
    return arr[0]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_array_sum(self):
    source = """fn main() -> i64:
    let arr: [i64; 5] = [1, 2, 3, 4, 5]
    let sum: i64 = 0
    for i in range(5):
        sum = sum + arr[i]
    return sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_array_len(self):
    source = """fn main() -> i64:
    let arr: [i64; 7] = [0, 0, 0, 0, 0, 0, 0]
    return arr.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_vec_push_and_access(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(10)
    nums.push(20)
    nums.push(30)
    return nums[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  def test_vec_len(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(1)
    nums.push(2)
    nums.push(3)
    return nums.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_vec_pop(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(5)
    nums.push(10)
    nums.push(15)
    let last: i64 = nums.pop()
    return last
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_vec_index_assign(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(1)
    nums.push(2)
    nums[0] = 100
    return nums[0]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_struct_basic(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.x + p.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_struct_field_assign(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 5, y: 10 }
    p.x = 100
    return p.x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_struct_multiple_fields(self):
    source = """struct Rectangle:
    x: i64
    y: i64
    width: i64
    height: i64

fn main() -> i64:
    let r: Rectangle = Rectangle { x: 0, y: 0, width: 10, height: 5 }
    return r.width * r.height
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 50

  def test_struct_pass_to_function(self):
    source = """struct Point:
    x: i64
    y: i64

fn sum_coords(px: i64, py: i64) -> i64:
    return px + py

fn main() -> i64:
    let p: Point = Point { x: 15, y: 25 }
    return sum_coords(p.x, p.y)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 40

  def test_comments(self):
    source = """# This is a comment at the top
fn main() -> i64:
    # Comment inside function
    let x: i64 = 10  # Inline comment
    # Another comment
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10

  def test_tuple_basic(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return t.0 + t.1
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_tuple_three_elements(self):
    source = """fn main() -> i64:
    let t: (i64, i64, i64) = (5, 10, 15)
    return t.0 + t.1 + t.2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_tuple_mixed_types(self):
    source = """fn main() -> i64:
    let t: (i64, bool) = (42, true)
    if t.1:
        return t.0
    return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_tuple_pass_elements(self):
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b

fn main() -> i64:
    let t: (i64, i64) = (15, 25)
    return add(t.0, t.1)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 40

  def test_enum_basic(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_enum_none_variant(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::None
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 99
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 99

  def test_enum_multiple_variants(self):
    source = """enum Result:
    Ok(i64)
    Err(i64)
    Unknown

fn main() -> i64:
    let r: Result = Result::Err(42)
    match r:
        Result::Ok(val):
            return val
        Result::Err(code):
            return code + 100
        Result::Unknown:
            return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 142

  def test_enum_in_function(self):
    source = """enum Option:
    Some(i64)
    None

fn unwrap_or(opt: Option, default: i64) -> i64:
    match opt:
        Option::Some(val):
            return val
        Option::None:
            return default

fn main() -> i64:
    let x: Option = Option::Some(10)
    let y: Option = Option::None
    let a: i64 = unwrap_or(x, 0)
    let b: i64 = unwrap_or(y, 99)
    return a + b
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 109  # 10 + 99

  def test_impl_basic(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_impl_with_args(self):
    source = """struct Counter:
    value: i64

impl Counter:
    fn add(self, n: i64) -> i64:
        return self.value + n

fn main() -> i64:
    let c: Counter = Counter { value: 100 }
    return c.add(42)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 142

  def test_impl_multiple_methods(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn get_x(self) -> i64:
        return self.x
    fn get_y(self) -> i64:
        return self.y
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    let p: Point = Point { x: 5, y: 15 }
    let a: i64 = p.get_x()
    let b: i64 = p.get_y()
    let c: i64 = p.sum()
    return a + b + c
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 40  # 5 + 15 + 20

  def test_impl_method_chain(self):
    source = """struct Value:
    n: i64

impl Value:
    fn get(self) -> i64:
        return self.n

fn main() -> i64:
    let v1: Value = Value { n: 10 }
    let v2: Value = Value { n: 20 }
    return v1.get() + v2.get()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30
