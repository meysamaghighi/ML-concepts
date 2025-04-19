# 🐍 Python Looping Cheatsheet

### 🔁 Basic Looping Patterns

| Type             | Purpose                  | Syntax                                     | Example Result                         |
|------------------|---------------------------|---------------------------------------------|----------------------------------------|
| **List**         | Loop over values          | `for x in lst:`                             | `10`, `20`, `30`                        |
| **List + index** | Index + value             | `for i, x in enumerate(lst):`              | `0 10`, `1 20`                          |
| **Tuple List**   | Unpack each tuple         | `for a, b in lst:`                          | `1 a`, `2 b`                            |
| **Dict**         | Loop over keys            | `for k in d:` or `for k in d.keys():`      | `'a'`, `'b'`                            |
| **Dict**         | Loop over values          | `for v in d.values():`                     | `1`, `2`                                |
| **Dict**         | Loop over key-values      | `for k, v in d.items():`                   | `'a' 1`, `'b' 2`                        |
| **Zip**          | Two lists together        | `for a, b in zip(l1, l2):`                 | `1 x`, `2 y`                            |
| **Iterator**     | Manual control            | `it = iter(lst)` + `next(it)`              | `1`, then `2`, etc.                     |

---

### 📋 Examples

#### List with Index
```python
lst = [10, 20, 30]
for i, x in enumerate(lst):
    print(i, x)
```

#### Dict Key + Value
```python
d = {'a': 1, 'b': 2}
for k, v in d.items():
    print(k, v)
```

#### List of Tuples
```python
pairs = [(1, 'a'), (2, 'b')]
for num, char in pairs:
    print(num, char)
```

#### Zipping Lists
```python
names = ['Alice', 'Bob']
scores = [95, 87]
for name, score in zip(names, scores):
    print(name, score)
```

---

### 🧠 Tips

- `enumerate()` is perfect when you need both index and value.
- `zip()` stops at the shortest list.
- `dict.items()` is the go-to for key-value looping.
- Prefer unpacking (`for a, b in ...`) when working with tuple-like structures.
