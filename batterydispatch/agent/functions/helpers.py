import warnings

def first_over(val, array):
    '''Returns the first value in the array that is equal to or over the given value val.
    The function assumes the array is sorted and the largest value is in the 'last' index. This is not checked.
    If the given value is greater than any value in the array, teh maximum array value is returned with a warning.'''
    if val > max(array):
        warnings.warn(
            "The value {} given is greater than the max value in array {}. The max value will be returned.".format(val,
                                                                                                                   array))
        return array[-1]
    first = next(a[1] for a in enumerate(array) if a[1] >= val)
    return first


def last_under(val, array):
    '''Returns the first value in the array that is less than the given value val.
    If all values in the array are smaller than the given value, the greatest (assumed last) value in the array is returned.
    The function assumes the array is sorted and the largest value is in the 'last' index. This is not checked.'''
    if val >= max(array):
        return array[-1]
    try:
        first_ix = next(a[0] for a in enumerate(array) if a[1] > val)
        if first_ix - 1 < 0:
            warnings.warn(
                "The value {} given is less than the first value in array {}. The min value will be returned.".format(
                    val, array))
            return array[0]
        else:
            return array[first_ix - 1]
    except StopIteration:
        raise StopIteration(
            "Unexpected StopIteration error raised looking for value {} in array {}.".format(val, array))
