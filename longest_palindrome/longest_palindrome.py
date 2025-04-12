class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        
        if n == 0:
            print("wrong input!")
            return ""

        def expand_from_center(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left + 1 : right]

        s_final = ""
        for i in range(n):
            # odd-length palindromes
            s_odd = expand_from_center(i, i)
            if len(s_odd) > len(s_final):
                s_final = s_odd
            
            # even-length palindromes
            s_even = expand_from_center(i, i+1)
            if len(s_even) > len(s_final):
                s_final = s_even

        return s_final
