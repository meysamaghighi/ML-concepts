# test_unittest.py

import unittest
from solution import Solution, list_from_digits, digits_from_list

class TestAddTwoNumbers(unittest.TestCase):
    def setUp(self):
        self.solution = Solution()

    def test_basic(self):
        l1 = list_from_digits([2, 4, 3])  # 342
        l2 = list_from_digits([5, 6, 4])  # 465
        result = digits_from_list(self.solution.addTwoNumbers(l1, l2))
        self.assertEqual(result, [7, 0, 8])  # 807

    def test_with_carry(self):
        l1 = list_from_digits([9, 9, 9])  # 999
        l2 = list_from_digits([1])       # 1
        result = digits_from_list(self.solution.addTwoNumbers(l1, l2))
        self.assertEqual(result, [0, 0, 0, 1])  # 1000

    def test_zeros(self):
        l1 = list_from_digits([0])  # 0
        l2 = list_from_digits([0])  # 0
        result = digits_from_list(self.solution.addTwoNumbers(l1, l2))
        self.assertEqual(result, [0])  # 0

    def test_unequal_lengths(self):
        l1 = list_from_digits([1])        # 1
        l2 = list_from_digits([9, 9])     # 99
        result = digits_from_list(self.solution.addTwoNumbers(l1, l2))
        self.assertEqual(result, [0, 0, 1])  # 100

if __name__ == '__main__':
    unittest.main()
