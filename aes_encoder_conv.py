import copy
import multiprocessing

import bitstring
import my_galios
import aes_toolkit as tk
import aes_key_expansion_256
import numpy as np


verbose = 0

def shift_rows(in_array):

    out_array = 16 * [0]

    out_array[0] = in_array[0]
    out_array[4] = in_array[4]
    out_array[8] = in_array[8]
    out_array[12] = in_array[12]

    out_array[1] = in_array[5]
    out_array[5] = in_array[9]
    out_array[9] = in_array[13]
    out_array[13] = in_array[1]

    out_array[2] = in_array[10]
    out_array[6] = in_array[14]
    out_array[10] = in_array[2]
    out_array[14] = in_array[6]

    out_array[3] = in_array[15]
    out_array[7] = in_array[3]
    out_array[11] = in_array[7]
    out_array[15] = in_array[11]

    return out_array



def mix_columns(in_array):

    out_array = 16 * [0]

    out_array[0] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[2, 3, 1, 1], y=[int(in_array[0], 2), int(in_array[1], 2), int(in_array[2], 2), int(in_array[3], 2)])}").bin
    out_array[4] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[2, 3, 1, 1], y=[int(in_array[4], 2), int(in_array[5], 2), int(in_array[6], 2), int(in_array[7], 2)])}").bin
    out_array[8] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[2, 3, 1, 1], y=[int(in_array[8], 2), int(in_array[9], 2), int(in_array[10], 2), int(in_array[11], 2)])}").bin
    out_array[12] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[2, 3, 1, 1], y=[int(in_array[12], 2), int(in_array[13], 2), int(in_array[14], 2), int(in_array[15], 2)])}").bin

    out_array[1] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 2, 3, 1], y=[int(in_array[0], 2), int(in_array[1], 2), int(in_array[2], 2), int(in_array[3], 2)])}").bin
    out_array[5] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 2, 3, 1], y=[int(in_array[4], 2), int(in_array[5], 2), int(in_array[6], 2), int(in_array[7], 2)])}").bin
    out_array[9] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 2, 3, 1], y=[int(in_array[8], 2), int(in_array[9], 2), int(in_array[10], 2), int(in_array[11], 2)])}").bin
    out_array[13] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 2, 3, 1], y=[int(in_array[12], 2), int(in_array[13], 2), int(in_array[14], 2), int(in_array[15], 2)])}").bin


    out_array[2] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 1, 2, 3], y=[int(in_array[0], 2), int(in_array[1], 2), int(in_array[2], 2), int(in_array[3], 2)])}").bin
    out_array[6] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 1, 2, 3], y=[int(in_array[4], 2), int(in_array[5], 2), int(in_array[6], 2), int(in_array[7], 2)])}").bin
    out_array[10] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 1, 2, 3], y=[int(in_array[8], 2), int(in_array[9], 2), int(in_array[10], 2), int(in_array[11], 2)])}").bin
    out_array[14] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[1, 1, 2, 3], y=[int(in_array[12], 2), int(in_array[13], 2), int(in_array[14], 2), int(in_array[15], 2)])}").bin

    out_array[3] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[3, 1, 1, 2], y=[int(in_array[0], 2), int(in_array[1], 2), int(in_array[2], 2), int(in_array[3], 2)])}").bin
    out_array[7] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[3, 1, 1, 2], y=[int(in_array[4], 2), int(in_array[5], 2), int(in_array[6], 2), int(in_array[7], 2)])}").bin
    out_array[11] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[3, 1, 1, 2], y=[int(in_array[8], 2), int(in_array[9], 2), int(in_array[10], 2), int(in_array[11], 2)])}").bin
    out_array[15] = bitstring.BitArray(f"uint8={my_galios.aes_dot_product(x=[3, 1, 1, 2], y=[int(in_array[12], 2), int(in_array[13], 2), int(in_array[14], 2), int(in_array[15], 2)])}").bin

    return out_array

    """
    15
    11
    7
    3
    
    14
    10
    6
    2
    
    13
    9
    5
    1
    
    12
    8
    4
    0
    
    
    
    """


def aes_round(state_array, key_array, round_idx):

    # print('\nbefore sub bytes',state_array)

    state_array = tk.get_sub_bytes(state_array)

    # print('after sub bytes', tk.bin_list_to_hex_string(state_array))


    if (verbose> 1): print(f'round[ {round_idx+1}].s_box\t{tk.bin_list_to_hex_string(state_array)}')

    # print('before shift rows',state_array)

    state_array = shift_rows(state_array)
    # print('after shift row', tk.bin_list_to_hex_string(state_array))


    if (verbose> 1): print(f'round[ {round_idx+1}].s_row\t{tk.bin_list_to_hex_string(state_array)}')


    # print('after shift row', state_array)







    # if round_idx != tk.conf['rounds_count'] - 1:

    state_array = mix_columns(state_array)
    # print('after column mix', state_array)

    if (verbose> 1): print(f'round[ {round_idx+1}].s_col\t{tk.bin_list_to_hex_string(state_array)}')






    # print('after col mix  ', tk.bin_list_to_hex_string(state_array))

    output_state_array = tk.xor_two_arrays(state_array, key_array)

    if (verbose> 1): print(f'round[ {round_idx + 2}].start\t{tk.bin_list_to_hex_string(output_state_array)}')

    return output_state_array


def aes_encrypt_128_bits(data_bin, keys_arrays):

    state_array = data_bin

    # print('input data',state_array)


    if (verbose> 1): print(f'round[ 0].input\t{tk.bin_list_to_hex_string(state_array)}')

    s_1 = tk.xor_two_arrays(state_array, keys_arrays[0])

    if (verbose> 1): print(f'round[ 0].start\t{tk.bin_list_to_hex_string(s_1)}')

    # print('first key',keys_arrays[0])


    state_arrays = [s_1]

    # print('after xor with first key',s_1)


    for i in range(tk.conf['rounds_count']):
        # print(f'\nround {i}')
        next_state_array = aes_round(state_arrays[-1], keys_arrays[i + 1], i)
        state_arrays.append(next_state_array)



    return state_arrays[-1]











def generate_gcm_prexor_cipher_texts(data_block_count, keys_arrays, iv, counter):

    prexor_cipher_texts = []

    block_based_data =[]

    for i in range(data_block_count):

        iv_counter = iv + bitstring.BitArray(f"uint32={counter}").bin
        print(f'payload_idx: {0}, i: {i}, counter: {counter}')


        iv_counter = ''.join(tk.chunks(iv_counter, 8)[::-1])  # to match with the block-based memory structure


        # print(f'Y{counter - 1}: {tk.bits_to_hex_string(iv_counter)}')


        assert len(iv_counter) == 128

        iv_counter_bin_list = tk.bit_string_to_bit_list(iv_counter)[0]
        # for el in iv_counter_bin_list:
        #     block_based_data.append(el)

        iv_counter_enctypted = aes_encrypt_128_bits(iv_counter_bin_list, keys_arrays)

        # print(f'E(K, Y{counter - 1}): {tk.bin_list_to_hex_string(iv_counter_enctypted)}')

        prexor_cipher_texts.append(iv_counter_enctypted)


        counter = tk.gcm_counter(counter)



        # counter = counter + 1


    return prexor_cipher_texts


def zero_pad_bin_list_to_multiples_of_128(input):

    if input == []:
        data_chunks = 16 * ['00000000']
    else:
        data_chunks = [input[i:i + 16] for i in range(0, len(input), 16)]

        if len(data_chunks[-1]) < 16:
            data_chunks[-1] = data_chunks[-1] + (16 - len(data_chunks[-1])) * ['00000000']

    return data_chunks

def aes_256_gcm_encrypt(data_bin, auth_bin, keys_arrays, iv, nuance, mode='encrypt'):

    data_bin_chunks = zero_pad_bin_list_to_multiples_of_128(data_bin)

    data_block_count = len(data_bin_chunks)


    # print(f"data_block_count: {data_block_count}")

    counter = tk.gcm_counter(nuance)


    # generate H
    H_bin_list = tk.bit_string_to_bit_list(128*'0')[0]
    H = aes_encrypt_128_bits(H_bin_list, keys_arrays)

    print(f'H: {tk.bin_list_to_hex_string(H)}')

    # generate the initial

    iv_counter_0 = iv + bitstring.BitArray(f"uint32={counter}").bin
    print('iv_counter_0 counter', counter)

    counter = tk.gcm_counter(counter)

    iv_counter_0 = ''.join(tk.chunks(iv_counter_0, 8)[::-1]) # to match with the block-based memory structure


    # print(f'Y0: {tk.bits_to_hex_string(iv_counter_0)}')


    iv_counter_0_bin_list = tk.bit_string_to_bit_list(iv_counter_0)[0]



    iv_counter_0_enctypted = aes_encrypt_128_bits(iv_counter_0_bin_list, keys_arrays)

    print(f'E(K, Y0): {tk.bin_list_to_hex_string(iv_counter_0_enctypted)}\n')

    # generate the cipher text without XOR

    prexor_cipher_texts = generate_gcm_prexor_cipher_texts(data_block_count, keys_arrays, iv, counter)


    print(tk.bin_list_to_hex_string(H),tk.bin_list_to_hex_string(iv_counter_0_enctypted),[tk.bin_list_to_hex_string(x) for x in prexor_cipher_texts])


    # adding XOR to the prexor_cipher texts
    cipher_texts = []

    for prexor_cipher_text, data_bin_chunk in zip(prexor_cipher_texts, data_bin_chunks):
        cipher_text = tk.xor_two_arrays(prexor_cipher_text, data_bin_chunk)
        cipher_texts.append(cipher_text)













    """
    building auth_tag after the cipher texts are at hand
    """

    auth_bin = zero_pad_bin_list_to_multiples_of_128(auth_bin)[0]


    auth_tag = my_galios.H_mult(auth_bin, H)

    # print(f"\nauth_tag_after_first_mult = {tk.bin_list_to_hex_string(auth_tag)}")


    # print("\nstarting the loop")

    # applying cipher texts
    for i in range(data_block_count):
        if mode == 'encrypt':
            auth_tag = tk.xor_two_arrays(auth_tag, cipher_texts[i])
        else:
            auth_tag = tk.xor_two_arrays(auth_tag, data_bin_chunks[i])

        # print(f"auth_tag_{i+1} after xor  = {tk.bin_list_to_hex_string(auth_tag)}")

        auth_tag = my_galios.H_mult(auth_tag, H)
        # print(f"auth_tag_{i+1} after mult = {tk.bin_list_to_hex_string(auth_tag)}")

    # print("end of the loop\n")


    # applying lengths
    A_bitstring = tk.bit_list_to_bit_string(auth_tag)
    C_bitstring = tk.bit_list_to_bit_string(cipher_texts)


    len_A = len(A_bitstring)
    len_C = len(C_bitstring)

    lens_bitstring = bitstring.BitArray(f"uint64={len_A}").bin + bitstring.BitArray(f"uint64={len_C}").bin
    lens_bin_list = tk.bit_string_to_bit_list(lens_bitstring)

    # print(f"len(A) || len(C)      = {tk.bits_to_hex_string(lens_bitstring)}")


    auth_tag = tk.xor_two_arrays(auth_tag, lens_bin_list[0])
    # print(f"auth_tag after xor with lens         = {tk.bin_list_to_hex_string(auth_tag)}")

    # the last mult H
    auth_tag = my_galios.H_mult(auth_tag, H)
    # print(f"auth_tag after last mult             = {tk.bin_list_to_hex_string(auth_tag)}")


    # xor with the initial iv_counter
    auth_tag = tk.xor_two_arrays(auth_tag, iv_counter_0_enctypted)
    # print(f"auth_tag after xor with iv_counter_0 = {tk.bin_list_to_hex_string(auth_tag)}")



    return cipher_texts, auth_tag


def main_conventional():

    with open("aes_256_key.txt", 'r') as my_file:
        key_hex =  my_file.read()

    with open("aes_256_data.txt", 'r') as my_file:
        data_hex =  my_file.read()


    naunce = 193939
    # iv = ''.join(np.random.choice(['0', '1'], 96))
    # iv = tk.hex_string_to_bit_string('cafebabefacedbaddecaf888')
    iv = tk.hex_string_to_bit_string('000000000000000000000000')

    assert len(iv) == 96

    auth_hex= ""

    data_bin = tk.hex_string_to_binary_list(data_hex)
    key_bin = tk.hex_string_to_binary_list(key_hex)
    auth_bin = tk.hex_string_to_binary_list(auth_hex)

    keys_arrays, keys_arrays_hex =  aes_key_expansion_256.expand_keys(key_bin)



    # debugging
    # ret = generate_gcm_prexor_cipher_texts(32, keys_arrays, iv, -1)
    #
    # sdf=[]
    # for el in ret:
    #     for x in el:
    #         sdf.append(x)
    #
    # hh=tk.convert_block_based_to_register_based_memory(sdf)
    # vv=4





    """
    GCM
    """



    cipher_texts, auth_tag = aes_256_gcm_encrypt(data_bin=data_bin, auth_bin=auth_bin, keys_arrays=keys_arrays, iv=iv, nuance=naunce, mode='encrypt')

    cipher_texts_flat = []
    for block in cipher_texts:
        for el in block:
            cipher_texts_flat.append(el)

    # manupulating the cipher text
    # a=copy.deepcopy(cipher_texts_flat)
    # tk.randomly_manupulate_one_bit_in_bin_list(cipher_texts_flat)
    # assert not a==cipher_texts_flat

    plain_texts, auth_tag_rec = aes_256_gcm_encrypt(data_bin=cipher_texts_flat, auth_bin=auth_bin, keys_arrays=keys_arrays, iv=iv, nuance=naunce, mode='decrypt')




    plain_texts_hex = tk.bin_list_to_hex_string(plain_texts)


    assert auth_tag == auth_tag_rec, f"auth_tag: {tk.bin_list_to_hex_string(auth_tag)} not the same as auth_tag_rec: {tk.bin_list_to_hex_string(auth_tag_rec)}"


    assert data_hex == plain_texts_hex



    print("\nauth_tag:")
    print(tk.bin_list_to_hex_string(auth_tag))



    print("\ncipher text:")
    for el in cipher_texts:
        print(tk.bin_list_to_hex_string(el))






    """
    single example
    """
    data_bin = data_bin[:16]
    # data_bin = data_bin[16: 32]

    encoded = aes_encrypt_128_bits(data_bin, keys_arrays)

    encoded = [bitstring.BitArray(f"uint8={int(x,2)}").hex for x in encoded]

    if (verbose> 1): print(encoded)
    # print("".join(encoded))
    # assert "".join(encoded) == "8ea2b7ca516745bfeafc49904b496089"

    print('all good')

    """
    e69fe5f4bb6696886ed019d59f622fef
    e69fe5f4bb6696886ed019d59f622fef
    
    
    08723272e5d1314cb8b881a52351040e
    08723272e5d1314cb8b881a52351040e
    """




if __name__ == '__main__':

    main_conventional()
