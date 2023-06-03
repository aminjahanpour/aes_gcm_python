import copy

import numpy as np

import aes_toolkit as tk
from numpy.polynomial import polynomial as poly
import bitstring

# irreducible_poly = 0b100011011             # mpy modulo x^8+x^4+x^3+x+1 (aes)
# irreducible_poly = 0b101100011               # mpy modulo x^8+x^6+x^5+x+1 (jaun example)

gcm_aes_degree = 128
gcm_aes_irreducible_poly = bitstring.BitArray(f"uint{gcm_aes_degree + 1}={(1 + (1<<1) + (1<<2) + (1<<7) + (1<<gcm_aes_degree))}").uint


def add(x, y):                  # add is xor
    return x^y

def sub(x, y):                  # sub is xor
    return x^y


def multiply_calculate(a, b, degree, irreducible_poly):

        if b > a:
            a, b = b, a

        c = 0
        while b > 0:
            if b & 0b1:
                c ^= a  # Add a(x) to c(x)

            b >>= 1  # Divide b(x) by x
            a <<= 1  # Multiply a(x) by x
            if a >= 2**degree:
                a ^= irreducible_poly  # Compute a(x) % p(x)

        return c



def mpy(x, y, degree=8, irreducible_poly=0b100011011):
    init_y = copy.deepcopy(y)

    m = 0

    for i in range(degree):

        m = m << 1

        # bitstring.BitArray(f"uint9={m}").bin , bitstring.BitArray(f"bin={100000000}").bin
        # if m & (0b100000000):
        if m & (1 << degree):
            m = m ^ irreducible_poly


        # if y & 0b010000000:
        if y & (1 << (degree-1)):
            m = m ^ x

        y = y << 1


    # print(f'time:                  768, a[          0]:   {x},    b[          0]:  {init_y},    result[          0]: {m}')

    return m

def div(x, y):                   # divide using inverse
    return mpy(x, inv(y))        #  (no check for y = 0)

def inv(x):                      # x^254 = 1/x
    p=mpy(x,x)                   # p = x^2
    x=mpy(p,p)                   # x = x^4
    p=mpy(p,x)                   # p = x^(2+4)
    x=mpy(x,x)                   # x = x^8
    p=mpy(p,x)                   # p = x^(2+4+8)
    x=mpy(x,x)                   # x = x^16
    p=mpy(p,x)                   # p = x^(2+4+8+16)
    x=mpy(x,x)                   # x = x^32
    p=mpy(p,x)                   # p = x^(2+4+8+16+32)
    x=mpy(x,x)                   # x = x^64
    p=mpy(p,x)                   # p = x^(2+4+8+16+32+64)
    x=mpy(x,x)                   # x = x^128
    p=mpy(p,x)                   # p = x^(2+4+8+16+32+64+128)
    return p


def aes_dot_product(x, y):
    ret = add(     mpy(x[0], y[0], degree=8, irreducible_poly=0b100011011),
                   mpy(x[1], y[1], degree=8, irreducible_poly=0b100011011))

    ret = add(ret, mpy(x[2], y[2], degree=8, irreducible_poly=0b100011011))
    ret = add(ret, mpy(x[3], y[3], degree=8, irreducible_poly=0b100011011))

    return ret


def H_mult(x, y):

    x_decimal = tk.bin_list_to_integer(x)
    y_decimal = tk.bin_list_to_integer(y)


    mult_ret = mpy(x_decimal, y_decimal, degree=gcm_aes_degree, irreducible_poly=gcm_aes_irreducible_poly)

    return tk.int_to_bin_list(mult_ret)


def s_box_circuit():
    a=0xAF # 1010 1111
    c = inv(a)

    c = bitstring.BitArray(f"uint8={c}").bin
    c = [int(x) for x in c]

    M = np.asarray([
        [1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,0],
        [0,0,0,1,1,1,1,1],
        [1,0,0,0,1,1,1,1],
        [1,1,0,0,0,1,1,1],
        [1,1,1,0,0,0,1,1],
        [1,1,1,1,0,0,0,1]
    ])

    # b = [1, 0, 1, 0, 1, 1, 1, 1]  # AF
    b = [0, 1, 1, 0, 0, 0, 1, 1]  # 63

    s= M @ c

    for i in range(8):
        s[i] = s[i] % 2


    ret = [0, 0, 0, 0, 0, 0, 0, 0]  # 1111001, 121

    for i in range(8):
        ret[i] = s[i] ^ b[i]  #b[(i + 4) % 8] ^ b[(i + 5) % 8] ^ b[(i + 6) % 8] ^ b[(i + 7) % 8] ^ st[i]

    print(ret)


def mult_by_two_GF_2_8(a):
    x = [int(d) for d in bitstring.BitArray(f'uint8={a}').bin][::-1]
    ret = 8 * [0]
    carry = x[7]

    ret[7] =  x[6]
    ret[6] =  x[5]
    ret[5] =  x[4]
    ret[4] = (x[3]) ^ carry
    ret[3] = (x[2]) ^ carry
    ret[2] =  x[1]
    ret[1] = (x[0]) ^ carry
    ret[0] = carry

    return int(''.join([str(x) for x in ret[::-1]]), 2)



def register_mult():
    pass

    ee = [

        [(0, 0), (1, 1), (2, 2), (3, 3)],
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        [(0, 2), (1, 3), (2, 0), (3, 1)],
        [(0, 3), (1, 0), (2, 1), (3, 2)],
        [(1, 1), (2, 2), (3, 3), (0, 0)],
        [(1, 2), (2, 3), (3, 0), (0, 1)],
        [(1, 3), (2, 0), (3, 1), (0, 2)],
        [(1, 0), (2, 1), (3, 2), (0, 3)],
        [(2, 2), (3, 3), (0, 0), (1, 1)],
        [(2, 3), (3, 0), (0, 1), (1, 2)],
        [(2, 0), (3, 1), (0, 2), (1, 3)],
        [(2, 1), (3, 2), (0, 3), (1, 0)],
        [(3, 3), (0, 0), (1, 1), (2, 2)],
        [(3, 0), (0, 1), (1, 2), (2, 3)],
        [(3, 1), (0, 2), (1, 3), (2, 0)],
        [(3, 2), (0, 3), (1, 0), (2, 1)],
    ]

    for e in ee:
        ret= []
        for el in e:
            i, j = el
            ret.append(32 * i + 8 * j)

        print(ret)




    for i in range(4): # row
        for j in range(4): # col
            index = 32 * i + 8 * j

            print(f'(i, j): ({i}, {j}) index = {index},     registers: {[index + x for x in range(8)]}')
        print("")

if __name__ == '__main__':
    asf = mult_by_two_GF_2_8(130)
    register_mult()
    s_box_circuit()

    a = 2
    b = 254

    print(multiply_calculate(a=a, b=b, degree=8, irreducible_poly=0b100011011))
    print(mpy(a, b, degree=8, irreducible_poly=0b100011011))


