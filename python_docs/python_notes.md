# Python Study Notes

- Use floor division for integer division: `//`
- `int` supports big integers natively
- Recursion limit is ~1000 calls
- Use strings for safer digit handling (splitting, summing)
- Handle edge cases (0, null, empty, etc)
- Set in python: `seen = set()`, `seen.add(c)`, `if c in seen:`
- If problem has recursive solution, it probably also has two-pointer or sliding window solution.
- `for i in range(n):` has a range of `0` to `n-1`
- `s[0]` is the first character in `str s`
- Substrings:
```python
s = "abcdefg"
print(s[1:5])  # Output: 'bcde'
```
- Return `""` for error of a function that returns string (not `None`, `null`, `-1`, etc.)
- DRY: Don't Repeat Yourself -- extract to helper functions
- Best algorithm for longeset palindromic substring is Manacher's algorithm in O(n) time.
- Type hints in python *does not* enforce types at runtime -- it's for documentation, readability, and tooling. Use them when using `mypy`, VS Code intellisense, production code, etc. Not needed for prototyping and quick scripts.
```python
def f(a: int, b: int) -> int:
    return a + b
```
- Array rotation w/ O(1) space: use reverse function.
- Sort list:
```python
nums: List[int]
nums.sort()
```
- Use `tuple` to add lists to sets. Use `sorted` to handle duplicates: `triplet = tuple(sorted([s, nums[low], nums[high]]))`
- Matrix is usually represented as a `List[List[int]]` list of lists. `matrix[i][j]` gives row i, column j.
- Count letters in a string: `s.count(c)`.
- Do k:th factor in `O(sqrt(n))` time.
- Convert string to lower case: `s = s.lower()`
- Check if string/characters are alphanumeric:
    - `c.isalnum()`
    - `s.isalnum()`
- Reverse of a string/list `s`--> `s[::-1]`
- **Maximum subarray problem** is solved by **Kadane's algorithm** in O(n). Idea is to keep current_sum and best_sum, and condition on whether j belongs to best found from 1 to j.
```python
def max_subarray(numbers):
    """Find the largest sum of any contiguous subarray."""
    best_sum = float('-inf')
    current_sum = 0
    for x in numbers:
        current_sum = max(x, current_sum + x)
        best_sum = max(best_sum, current_sum)
    return best_sum
```
- Subset sum problem (does S have a subset w/ sum X?) is NP-complete.
- **Stack** in python is list:
```python
stack = []
# push
stack.append(10)
# pop
top = stack.pop()
# peek
top = stack[-1]
# is empty
empty = len(stack) == 0
```
- 