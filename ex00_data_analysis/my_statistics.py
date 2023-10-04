import numpy as np


def ft_count(nums: np.array):
    """Count the number of items in array"""
    count = 0
    for num in nums:
        count += 1
    return float(count)


def ft_min(nums: np.array):
    """Return min element"""
    return sorted(nums)[0]


def ft_max(nums: np.array):
    """Return max element"""
    return sorted(nums)[-1]


def ft_total(nums: np.array):
    """Compute the total sum"""
    if not check_input(nums):
        return None
    count = 0
    for num in nums:
        count += num
    return count


def ft_mean(nums: np.array):
    """Compute the mean"""
    if not check_input(nums):
        return None
    count = ft_total(nums)
    mean = count / len(nums)
    return mean


def ft_median(nums: np.array):
    """Compute the median"""
    if not check_input(nums):
        return None
    sort_nums = sorted(nums)
    size = len(nums)
    if size % 2:
        median = sort_nums[size // 2]
    else:
        median = (sort_nums[size // 2] + sort_nums[size // 2 - 1]) / 2
    return median


def ft_quartile3(nums: np.array):
    """Compute the Q3 quartiles"""
    if not check_input(nums):
        return None
    N = len(nums)
    sort_nums = sorted(nums)

    q3_ind = (3 * N + 1) / 4 - 1
    if q3_ind.is_integer():
        q3 = sort_nums[int(q3_ind)]
    elif (q3_ind - 0.25).is_integer():
        q3 = (sort_nums[int(q3_ind - 0.25)] * 3
              + sort_nums[int(q3_ind + 0.75)] * 1) / 4
    elif (q3_ind - 0.75).is_integer():
        q3 = (sort_nums[int(q3_ind + 0.25)] * 3
              + sort_nums[int(q3_ind - 0.75)] * 1) / 4
    elif (q3_ind - 0.5).is_integer():
        q3 = (sort_nums[int(q3_ind - 0.5)] + sort_nums[int(q3_ind + 5)]) / 2

    return q3


def ft_quartile1(nums: np.array):
    """Compute the Q1 quartile"""
    if not check_input(nums):
        return None
    N = len(nums)
    sort_nums = sorted(nums)

    q1_ind = (N + 3) / 4 - 1
    if q1_ind.is_integer():
        q1 = sort_nums[int(q1_ind)]
    elif (q1_ind - 0.25).is_integer():
        q1 = (sort_nums[int(q1_ind - 0.25)] * 3
              + sort_nums[int(q1_ind + 0.75)] * 1) / 4
    elif (q1_ind - 0.75).is_integer():
        q1 = (sort_nums[int(q1_ind + 0.25)] * 3
              + sort_nums[int(q1_ind - 0.75)] * 1) / 4
    elif (q1_ind - 0.5).is_integer():
        q1 = (sort_nums[int(q1_ind - 0.5)] + sort_nums[int(q1_ind + 0.5)]) / 2

    return q1


def ft_std(nums: np.array):
    """Compute standard deviation"""
    if not check_input(nums):
        return None
    var = ft_var(nums)
    return (var ** 0.5)


def ft_var(nums: np.array):
    """Compute variance"""
    if not check_input(nums):
        return None
    mean = ft_mean(nums)
    var = 0
    for num in nums:
        var += (num - mean) ** 2
    var /= len(nums)
    return var


def check_input(nums: np.array):
    """Verify args"""
    try:
        assert nums.size, "empty array."
        assert type(nums[0]) is np.single or type(nums[0]) is np.double,\
            "need a numpy array of single or double."
        return True
    except AssertionError as err:
        print(err)
        return False
