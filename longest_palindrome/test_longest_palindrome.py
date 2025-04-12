from longest_palindrome import Solution

def test_null():
    s = Solution()
    assert s.longestPalindrome("") == ""

def test_single():
    s = Solution()
    assert s.longestPalindrome("a") == "a"

def test_two():
    s = Solution()
    assert s.longestPalindrome("ab") == "a"

def test_three():
    s = Solution()
    assert s.longestPalindrome("bab") == "bab"
    assert s.longestPalindrome("aba") == "aba"

def test_four():
    s = Solution()
    assert s.longestPalindrome("cbbd") == "bb"

def test_five():
    s = Solution()
    assert s.longestPalindrome("babad") == "bab" or s.longestPalindrome("babad") == "aba"

def test_seven():
    s = Solution()
    assert s.longestPalindrome("aaca325kacaa") == "aca"  