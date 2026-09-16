from collections import Counter
from collections import defaultdict
from typing import Dict


def combine_dict_of_list(*args) -> Dict:
    """Combine Dict of list

    Args:
        *args: arguments

    Returns:
        :py:class:`Dict`: Dict of Combined list
    """
    result = defaultdict(list)

    for arg in args:
        for key, vals in arg.items():
            result[key].extend(vals)

    return dict(result)


def combine_dict_of_int(*args) -> Dict:
    """combine Dict of Integer

    Args:
        *args: arguments

    Returns:
        :py:class:`Dict`: Dict of combined integer
    """
    result = Counter()
    for arg in args:
        result += Counter(arg)

    return dict(result)


def combine_dict_of_dict(*args) -> Dict:
    """combine Dict of Dict

    Args:
        *args: dictionaries

    Returns:
        :py:class:`Dict`: Dict of combined Dictionary
    """
    result = defaultdict(dict)
    for arg in args:
        for key, data in arg.items():
            result[key].update(data)

    return result


def merge_default_dict(obj1, obj2):
    """
    Dictionary merge by updating but not overwriting if value exists
    Merge obj1 with obj2
    """

    output = obj1.copy()
    for key, val in obj2.items():
        if key not in output:
            output[key] = val
        elif type(val) == dict:
            output[key] = merge_default_dict(output[key], val)
        else:
            # Attribute value exists
            pass

    return output


def merge_dicts(*args) -> Dict:
    """merge dict into one

    Args:
        *args: input dictionary

    Returns:
        :py:class:`Dict`: merged dict
    """
    result = {}
    for arg in args:
        result.update(arg)

    return dict(result)
