import pytest
from longest_palindrome import Solution

@pytest.mark.parametrize("input_str, expected_outputs", [
    ("", [""]),
    ("a", ["a"]),
    ("ab", ["a", "b"]),
    ("bab", ["bab"]),
    ("aba", ["aba"]),
    ("cbbd", ["bb"]),
    ("babad", ["bab", "aba"]),
    ("aaca325kacaa", ["aca"]),
])
def test_longest_palindrome_param(input_str, expected_outputs):
    s = Solution()
    result = s.longestPalindrome(input_str)
    assert result in expected_outputs, f"For input '{input_str}', got '{result}' which is not in {expected_outputs}"
