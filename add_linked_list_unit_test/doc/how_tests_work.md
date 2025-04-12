# 🧪 How Testing Works (unittest & pytest)

This guide explains how `pytest` and `unittest` discover and run tests in this repo.

---

## ✅ How `pytest` Works

### 🔍 Test Discovery Rules:
- `pytest` automatically finds and runs:
  - Any **file** that starts with `test_` or ends with `_test.py`
  - Any **function** that starts with `test_`
  - Any **class** that starts with `Test`, even without inheriting from a base class (like `unittest.TestCase`)
  - Any method inside that class that starts with `test_`

### 🧠 Notes:
- You do **not** need to inherit from any class.
- Fixtures (`@pytest.fixture`) and asserts are native — no need for special methods like `.assertEqual()`.

### ▶️ Example:

```python
# pytest will discover this function
def test_sum():
    assert 2 + 2 == 4
```

To run all tests:
```bash
pytest .
```

To run a specific test file:
```bash
pytest test_pytest.py
```

---

## ✅ How `unittest` Works

### 🔍 Test Discovery Rules:
- You must create a class that **inherits from `unittest.TestCase`**
- All **methods** inside that class that start with `test_` will be run
- You must call `unittest.main()` to kick off the test run

### 🧠 Notes:
- Assertions are done using methods like `.assertEqual()`, `.assertTrue()`, etc.
- You can use a `setUp()` method to initialize objects for each test.

### ▶️ Example:

```python
import unittest

class TestMath(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 2, 3)

if __name__ == '__main__':
    unittest.main()
```

To run:
```bash
python test_unittest.py
```

---

## ✅ Summary Table

| Feature               | `pytest`                    | `unittest`                 |
|-----------------------|-----------------------------|----------------------------|
| File naming           | `test_*.py` or `*_test.py`  | Any name, typically `test_` |
| Function naming       | `test_*`                    | `test_*` inside a class     |
| Class base            | Optional                    | Must inherit `unittest.TestCase` |
| Assertion style       | Native `assert`             | `.assertEqual()`, etc.      |
| Setup method          | `@pytest.fixture` or `setup_method` | `setUp()`            |
| Run command           | `pytest`                    | `python test_unittest.py`   |

