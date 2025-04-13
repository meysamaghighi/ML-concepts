class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False

    def get_balance(self):
        return self.balance


def welcome_message():
    return "Welcome to Python Bank!"


def main():
    print(welcome_message())
    
    # Creating an account
    account = BankAccount("Meysam", 100)
    print(f"Initial balance: {account.get_balance()}")

    # Deposit
    account.deposit(50)
    print(f"After deposit: {account.get_balance()}")

    # Withdraw
    if account.withdraw(30):
        print(f"After withdrawal: {account.get_balance()}")
    else:
        print("Withdrawal failed.")

if __name__ == "__main__":
    main()
