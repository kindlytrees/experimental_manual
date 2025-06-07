
# python -m doctest doctests.py -v
def sum_evens(numList):
    """
        return sum of the even num in parameter named numList

    >>> sum_evens([1,2,3,4,5,6,7,8,9,10])
    30

    >>> sum_evens([2,3,4,5,6])
    11
    """
    result = 0
    for idx, value in enumerate(numList):
        if value%2 == 0 :
            result += value
    return result

