# solution.py

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        n1 = self.list_to_int(l1)
        n2 = self.list_to_int(l2)
        return self.int_to_list(n1 + n2)

    def list_to_int(self, l: Optional[ListNode]) -> int:
        multiplier = 1
        num = 0
        while l:
            num += l.val * multiplier
            multiplier *= 10
            l = l.next
        return num

    def int_to_list(self, n: int) -> Optional[ListNode]:
        if n == 0:
            return ListNode(0)

        dummy = ListNode()
        current = dummy

        while n > 0:
            digit = n % 10
            current.next = ListNode(digit)
            current = current.next
            n = n // 10

        return dummy.next


# Helper functions to use in test files
def list_from_digits(digits):
    head = current = ListNode(digits[0])
    for digit in digits[1:]:
        current.next = ListNode(digit)
        current = current.next
    return head

def digits_from_list(l):
    digits = []
    while l:
        digits.append(l.val)
        l = l.next
    return digits
