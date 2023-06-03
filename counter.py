import numpy as np


def check_unique(lst):
    # use the unique function from numpy to find the unique elements in the list
    unique_elements, counts = np.unique(lst, return_counts=True)
    # return True if all elements in the list are unique (i.e., the counts are all 1)
    return all(counts == 1)



nuance = 193939

counter = nuance

values = []




for i in range(1_000_000_000):

    counter = (counter * 311) % (2 ** 32)

    values.append(counter)


# cc = nuance
# for i in range(1000):
#     cc += 1
#     carry = (~(cc | -cc)) >> 31;
#     cc += carry;
#     carry &= (~(cc | -cc)) >> 31;
#     cc += carry;
#     carry &= (~(cc | -cc)) >> 31;
#     cc += carry;
#
#     values.append(cc)
#




print(values[-100:])
print('values ready')
assert (len(set(values)) == len(values))
print(check_unique(values))
