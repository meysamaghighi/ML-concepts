# test_pytest.py

from solution import Solution, list_from_digits, digits_from_list

def test_basic():
    s = Solution()
    l1 = list_from_digits([2, 4, 3])  # 342
    l2 = list_from_digits([5, 6, 4])  # 465
    assert digits_from_list(s.addTwoNumbers(l1, l2)) == [7, 0, 8]  # 807

def test_with_carry():
    s = Solution()
    l1 = list_from_digits([9, 9, 9])  # 999
    l2 = list_from_digits([1])       # 1
    assert digits_from_list(s.addTwoNumbers(l1, l2)) == [0, 0, 0, 1]  # 1000

def test_zeros():
    s = Solution()
    l1 = list_from_digits([0])  # 0
    l2 = list_from_digits([0])  # 0
    assert digits_from_list(s.addTwoNumbers(l1, l2)) == [0]  # 0

def test_unequal_lengths():
    s = Solution()
    l1 = list_from_digits([1])        # 1
    l2 = list_from_digits([9, 9])     # 99
    assert digits_from_list(s.addTwoNumbers(l1, l2)) == [0, 0, 1]  # 100
