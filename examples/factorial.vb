fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main() -> i64:
    let result: i64 = factorial(5)
    print(result)
    return 0

