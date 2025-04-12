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
- 