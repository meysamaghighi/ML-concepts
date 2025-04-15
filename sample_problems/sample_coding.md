### **1. Count Pairs with Sum Less Than K**

#### **Problem Description**  
Given a list of integers `nums` and an integer `K`, return the number of pairs `(i, j)` such that `i < j` and `nums[i] + nums[j] < K`.

---

#### **Brute Force Solution (O(n²))**  
We check every possible pair to see if their sum is less than `K`.

```python
def count_pairs_brute_force(nums, K):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] < K:
                count += 1
    return count
```

---

#### **Optimized Solution using Two Pointers (O(n log n))**  
We sort the list and use two pointers from both ends to count valid pairs more efficiently.

```python
def count_pairs_optimized(nums, K):
    nums.sort()
    left, right = 0, len(nums) - 1
    count = 0

    while left < right:
        if nums[left] + nums[right] < K:
            count += (right - left)
            left += 1
        else:
            right -= 1
    return count
```

---

### **2. Peak Load Interval (Login + Logout Times)**

#### **Problem Description**  
Given a list of login and logout time intervals like `[(2, 10), (5, 8)]`, find the **time interval** when the **maximum number of users** were online at the same time.

---

#### **Solution Using Event Timeline Sorting**  
We treat each login as `+1` and logout as `-1`, sort all events, and track the number of users online over time.

```python
def find_peak_load_interval(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))   # login event
        events.append((end, -1))    # logout event

    events.sort()
    max_users = current_users = 0
    peak_start = peak_end = None

    for i in range(len(events)):
        time, delta = events[i]
        current_users += delta
        if current_users > max_users:
            max_users = current_users
            peak_start = time
            # Next time the user count drops, we record peak_end
            if i + 1 < len(events):
                peak_end = events[i + 1][0]

    return (peak_start, peak_end)
```

---

### **3. Peak Load from Only Login Times**

#### **Problem Description**  
Given a list of login timestamps (e.g., `[2, 5, 5, 5, 7, 8]`), find the time when the most users were logged in. Each login is considered instantaneous or has a fixed session duration.

---

#### **Solution (Frequency Count)**  
We count how many users logged in at each time and return the time with the maximum count.

```python
from collections import defaultdict

def peak_load_from_logins(login_times):
    count = defaultdict(int)
    for time in login_times:
        count[time] += 1
    
    max_users = 0
    peak_time = None
    for time, users in count.items():
        if users > max_users:
            max_users = users
            peak_time = time

    return peak_time, max_users
```

---

### **4. Dot Product of Two Sparse Vectors**

#### **Problem Description**  
Given two sparse vectors, return their dot product. Each vector is represented as a dictionary of index-value pairs.

---

#### **Simple Efficient Solution (O(n))**  
Iterate over one dictionary and multiply only matching indices.

```python
def dot_product_sparse(vec1, vec2):
    return sum(vec1[i] * vec2[i] for i in vec1 if i in vec2)
```

---

#### **Optimized for Sparse `vec2` using Binary Search (O(m log n))**  
If `vec2` is very sparse, and `vec1` is sorted as a list of `(index, value)` pairs, use binary search.

```python
from bisect import bisect_left

def dot_product_sparse_bs(vec1, vec2):
    # vec1: list of (index, value) sorted by index
    # vec2: dict of sparse entries

    indices = [i for i, _ in vec1]
    values = {i: v for i, v in vec1}
    
    result = 0
    for i in vec2:
        pos = bisect_left(indices, i)
        if pos < len(indices) and indices[pos] == i:
            result += values[i] * vec2[i]
    return result
```

---

### **5. Unique Customers Who Visited Every Day**

#### **Problem Description**  
Given a log of website visits for 7 days (as a list of sets, each containing users who visited that day), return the set of users who visited **every day**.

---

#### **Solution Using Set Intersection**

```python
def customers_visited_every_day(logs_by_day):
    if not logs_by_day:
        return set()

    return set.intersection(*logs_by_day)
```

---

#### **Example Usage**

```python
logs_by_day = [
    {'alice', 'bob', 'carol'},
    {'alice', 'bob'},
    {'alice', 'bob', 'dave'},
    {'bob', 'alice'},
    {'bob', 'alice'},
    {'bob', 'alice', 'frank'},
    {'bob', 'alice'}
]

print(customers_visited_every_day(logs_by_day))  # Output: {'alice', 'bob'}
```
