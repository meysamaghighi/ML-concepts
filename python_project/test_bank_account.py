import pytest
from bank_account import BankAccount, welcome_message

def test_welcome_message():
    assert welcome_message() == "Welcome to Python Bank!"

def test_deposit():
    acc = BankAccount("Test", 100)
    acc.deposit(50)
    assert acc.get_balance() == 150

def test_withdraw_success():
    acc = BankAccount("Test", 100)
    result = acc.withdraw(40)
    assert result is True
    assert acc.get_balance() == 60

def test_withdraw_fail():
    acc = BankAccount("Test", 30)
    result = acc.withdraw(50)
    assert result is False
    assert acc.get_balance() == 30
