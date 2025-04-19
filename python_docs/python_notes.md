# Python Study Notes

- Must-visit overview: https://leetcode.com/explore/interview/card/cheatsheets/720/resources/4723/
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
- Merge sorted arrays in-place: start from the end!
- Use `OrderedDict` when you need insertion order in a dictionary (it's a subclass of `dict`). Use in LRU/LFU Cache, First Unique Element, Log System or counter.
    - **O(1)** access, update and deletion with maintained order.
```python
from collections import OrderedDict

od = OrderedDict()
od['a'] = 1
od['b'] = 2
od.move_to_end('a')     # Moves 'a' to the end
od.popitem(last=False)  # Removes 'b' (the first inserted key)
```
- `Counter` usage:
```python
>>> from collections import Counter
>>> c = Counter("banana")
>>> c
Counter({'a': 3, 'n': 2, 'b': 1})
>>> c.most_common(1)
[('a', 3)]
>>> c.most_common(2)
[('a', 3), ('n', 2)]
>>> c.most_common(1)[0][0]
'a'
```
- Strings are immutable:
```python
ans = "-" * len(s)
ans[2 * i] = char1  # ❌ won't work
```
Correct form:
```python
ans = [""] * len(s)
...
return ''.join(ans)
```
- Example w/ `regex`, `Counter`, and `lower()`:
```python
def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
    
    words = re.findall(r"\w+", paragraph.lower())
    b_words = [word.lower() for word in banned]
    count = collections.Counter(word for word in words if word not in b_words)
    
    return count.most_common(1)[0][0]
```
- Use `dict.get()` for simpler **None-handling**:
```python
x = my_map[key]     # ❌ doesn't handle key = None
x = my_map.get(key) # ✅ Handles key = None -> returns None```
```
- **Recursive solutions**:
    - Each recursive call adds a new **stack frame** to the call stack (storing local variables, return addresses, function metadata). This adds **additional O(n) space just for the call stack**!
    - Python has a recursion **depth limit of 1000**.
    - **Unpredictable garbage collection**.
- **Set intersection** and **unpacking a list**:
```python
def customers_visited_every_day(logs_by_day):
    if not logs_by_day:
        return set()

    return set.intersection(*logs_by_day)
```
- 