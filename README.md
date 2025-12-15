# Vibec

[![CI](https://github.com/markoelez/vibec/actions/workflows/ci.yml/badge.svg)](https://github.com/markoelez/vibec/actions/workflows/ci.yml)

A toy compiled programming language with Python/Rust hybrid syntax, targeting ARM64 macOS.

## Language Features

Vibec combines the visual cleanliness of Python's indentation-based blocks with Rust's explicit type annotations:

```
fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main() -> i64:
    let result: i64 = factorial(5)
    print(result)
    return 0
```

### Syntax

- `fn` keyword for functions with `-> ReturnType`
- `let` for variable declarations with explicit types
- Python-style indentation for blocks (no braces)
- Colons to start blocks

### Supported Types

- `i64` - 64-bit signed integer
- `bool` - boolean (`true` / `false`)

### Supported Constructs

- Functions with parameters and return values
- Variables with `let`
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`
- Control flow: `if`/`else`, `while`
- Built-in: `print()` for output

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

## Examples

### Hello World (exit code)

```
fn main() -> i64:
    return 42
```

```bash
$ vibec hello.vb && ./hello; echo $?
42
```

### Factorial

```
fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main() -> i64:
    print(factorial(5))
    return 0
```

```bash
$ vibec factorial.vb && ./factorial
120
```

### Fibonacci

```
fn fib(n: i64) -> i64:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fn main() -> i64:
    print(fib(10))
    return 0
```

```bash
$ vibec fibonacci.vb && ./fibonacci
55
```

## Architecture

The compiler is structured as a clean pipeline:

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

## Requirements

- Python 3.12+
- macOS with ARM64 (Apple Silicon)
- Xcode Command Line Tools (for `as` and `ld`)

## License

MIT

