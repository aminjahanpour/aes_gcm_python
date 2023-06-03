import numpy as np
import bitstring
import pandas as pd
from cipher_analysis import *


with open("aes_256_key.txt", 'r') as my_file:
    key_hex = my_file.read()

key_bin = tk.hex_string_to_bit_string(key_hex)

def get_integer_from_string(x):


    matching_bits = 0

    for idx, bit in enumerate(x):
        if bit == key_bin[256 - key_missing_bits_count + idx]:
            matching_bits += 1
    return matching_bits
    #
    # l = len(x)
    #
    # a =  bitstring.BitArray(f"bin={x}").uint & 0b1110100
    # b =  bitstring.BitArray(f"uint{l}={a}").bin
    #
    # return b
def main():
    df = pd.read_csv("./cost_function.log",names=['key', 'obj_1', 'obj_2'], dtype={'key': str, 'obj_1': np.float64, 'obj_2': np.float64})

    # df['key_similirity'] =  df.apply(get_integer_from_string,1,)

    df['key_match'] = df['key'].apply(get_integer_from_string)

    df.drop_duplicates(subset=['key'], inplace=True)

    i = df[(df.key == key_bin[-key_missing_bits_count:])].index
    df.drop(i, inplace=True)

    df.sort_values(by=['key_match'], inplace=True)

    sdf=4

if __name__ == '__main__':
    main()