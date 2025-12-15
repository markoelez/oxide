# Vibec

[![CI](https://github.com/markoelez/vibec/actions/workflows/ci.yml/badge.svg)](https://github.com/markoelez/vibec/actions/workflows/ci.yml)

A toy compiled programming language with Python/Rust hybrid syntax, targeting ARM64 macOS.

## Language Features

Vibec combines Python's indentation-based blocks with Rust's explicit type annotations:

```
# Enums with pattern matching
enum Option:
    Some(i64)
    None

# Structs
struct Point:
    x: i64
    y: i64

fn unwrap_or(opt: Option, default: i64) -> i64:
    match opt:
        Option::Some(val):
            return val
        Option::None:
            return default

fn main() -> i64:
    # Enums and pattern matching
    let x: Option = Option::Some(42)
    let result: i64 = unwrap_or(x, 0)

    # Structs
    let p: Point = Point { x: 10, y: 20 }
    p.x = 100

    # Arrays, vectors, tuples
    let arr: [i64; 3] = [1, 2, 3]
    let nums: vec[i64] = []
    nums.push(42)
    let t: (i64, bool) = (100, true)

    # Control flow
    for i in range(5):
        print(i)

    return factorial(5)

fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**Supported:** enums with `match`, functions, structs, tuples `(T1, T2)`, arrays `[T; N]`, vectors `vec[T]`, `if`/`else`, `while`, `for i in range()`, arithmetic, comparisons, logical ops, `print()`

## Architecture

The compiler is structured as follows:

```
Source Code → Lexer → Parser → Type Checker → Code Generator → ARM64 Assembly
```

### Modules

- `tokens.py` - Token definitions
- `lexer.py` - Tokenization with Python-style indentation handling
- `ast.py` - AST node dataclasses
- `parser.py` - Recursive descent parser with precedence climbing
- `checker.py` - Type checking with scoped symbol tables
- `codegen.py` - ARM64 assembly generation for macOS
- `compiler.py` - Pipeline orchestration
- `cli.py` - Command-line interface


## Installation

```bash
# Clone the repository
git clone https://github.com/markoelez/vibec.git
cd vibec

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
# Compile a source file to executable
vibec source.vb

# Specify output file
vibec source.vb -o myprogram

# Output assembly only
vibec source.vb --emit-asm

# Keep assembly file alongside binary
vibec source.vb --keep-asm
```

## Requirements

- Python 3.12+
- macOS with ARM64 (Apple Silicon)
- Xcode Command Line Tools (for `as` and `ld`)

## License

MIT

