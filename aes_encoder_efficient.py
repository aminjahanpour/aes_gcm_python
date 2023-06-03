import copy
import multiprocessing
import shutil

import bitstring
import my_galios
import aes_toolkit as tk
import aes_key_expansion_256
import numpy as np
import aes_encoder_conv



verbose = 0



mix_col_ind = [
    [[0, 5, 10, 15], [5, 10, 15, 0], [10, 15, 0, 5], [15, 0, 5, 10]],
    [[4, 9, 14, 3], [9, 14, 3, 4], [14, 3, 4, 9], [3, 4, 9, 14]],
    [[8, 13, 2, 7], [13, 2, 7, 8], [2, 7, 8, 13], [7, 8, 13, 2]],
    [[12, 1, 6, 11], [1, 6, 11, 12], [6, 11, 12, 1], [11, 12, 1, 6]]
]


global used_counters
used_counters = []

"""
we work on N = 32 blocks of data simultaneously

So we have 32 blocks of 128-bit long data. This amounts to 512 bytes of data.






block structure:

byte 0:
bit 7   : bit 0

byte 1:
bit 15  : bit 8

...

byte 15:
bit 127 : bit 120




ram structure (block-based)

b(i, j, k):
    i: block index   [0 : 31]
    j: byte counter  [0 : 15]
    k: bit counter   [0 : 7 ]
    
    

    

block 0:

b(0 ,  0,  7), b(0 ,  0,  6), ... , b(0 ,  0,  0)           |       register 7  , register 6  , ... register   0
b(0 ,  1,  7), b(0 ,  1,  6), ... , b(0 ,  1,  0)           |       register 15 , register 14 , ... register   8

...
b(0 , 15,  7), b(0 , 15,  6), ... , b(0 , 15,  0)           |       register 127, register 126, ... register 120


...


block 31:


b(31,  0,  7), b(31,  0,  6), ... , b(31,  0,  0)           |       register 7  , register 6  , ... register   0
b(31,  1,  7), b(31,  1,  6), ... , b(31,  1,  0)           |       register 15 , register 14 , ... register   8

...
b(31, 15,  7), b(31, 15,  6), ... , b(31, 15,  0)           |       register 127, register 126, ... register 120





------------------------------------------------------------

ram structure (register-based)




register 0    (first bit of all blocks):

b(0 ,  0,  0), b(1 ,  0,  0), ..., b(7 ,  0,  0) 
b(8 ,  0,  0), b(9 ,  0,  0), ..., b(15,  0,  0) 
b(16,  0,  0), b(17,  0,  0), ..., b(23,  0,  0) 
b(24,  0,  0), b(25,  0,  0), ..., b(31,  0,  0) 



register 1     (2nd bit of all blocks):

b(0 ,  0,  1), b(1 ,  0,  1), ..., b(7 ,  0,  1) 
b(8 ,  0,  1), b(9 ,  0,  1), ..., b(15,  0,  1) 
b(16,  0,  1), b(17,  0,  1), ..., b(23,  0,  1) 
b(24,  0,  1), b(25,  0,  1), ..., b(31,  0,  1) 


...


register 8    (9th bit of all blocks):

b(0 ,  1,  0), b(1 ,  1,  0), ..., b(7 ,  1,  0) 
b(8 ,  1,  0), b(9 ,  1,  0), ..., b(15,  1,  0) 
b(16,  1,  0), b(17,  1,  0), ..., b(23,  1,  0) 
b(24,  1,  0), b(25,  1,  0), ..., b(31,  1,  0) 




register 9:   (10-th bit)

b(0 ,  1,  1), b(1 ,  1,  1), ..., b(7 ,  1,  1) 
b(8 ,  1,  1), b(9 ,  1,  1), ..., b(15,  1,  1) 
b(16,  1,  1), b(17,  1,  1), ..., b(23,  1,  1) 
b(24,  1,  1), b(25,  1,  1), ..., b(31,  1,  1) 



...





register 127:   (last bit of all blocks)

b(0 ,  1,  1), b(1 ,  1,  1), ..., b(7 ,  1,  1) 
b(8 ,  1,  1), b(9 ,  1,  1), ..., b(15,  1,  1) 
b(16,  1,  1), b(17,  1,  1), ..., b(23,  1,  1) 
b(24,  1,  1), b(25,  1,  1), ..., b(31,  1,  1) 







to map from block-based to register-based:
    - read the block-based memory byte by byte
    - for each byte:
        - identify the 8 registers that are to contain bits of the byte
            use a 16-counter
            counter -> registers: [counter * 8 + 0, counter * 8 + 1, counter * 8 + 2, ... , counter * 8 + 7]   

        - for each identified register:
            - identify the target row in the register memory
                target_row = register * 4 + block_number / 8
            - left shift the current value of the register
            - OR with the new bit




Every register contains the i_th bit of all the 32 data blocks.

So every register is 32-bit (4 bytes; one word) long and we have 128 registers in total.

new ram layout is set up to facilitate access to R[i, j]
where:
    i:  register id     0 <= i < 128
    j:  register index  0 <= j < 32

R[3 , 5] means
    3: the register that stores all the 3th bit of all the 32 blocks
    5: the fifth bit in that register
    
    this is the same as 3th bit of block number 5



we need to have convinient access to resigters by their `id`.
so we need to reformat the memory from its original block-based structure to a
register-based structure.







to perform sub_byte on all the bytes in parallel, we can not use memory 
because we can not simultaneously read from it or write to it from more than one 
process. we can't do that even for 2 processes. so we can not use the memory
if we are to employ parallel computation.

we choose to perform sub-byte for 8 registers at a time.
in total we have 128 registers.
so we need to perform 128 / 8 = 16 rounds.
at each round we read 8 full registers from the memory.
each register is 32-bit long (4 bytes; one word).

    number of sub-bytes instance in parallel      =        number of registers operated on in parallel    *     number of bytes in one register
                                            32    =    8   *   4

so we need to instantiate 32 instances of sub-byte by genvar.
this assumes we can afford to spend 8 * 32 = 256 flip-flops to feed the sub-byte at each round.


16 rounds:
    each round we take 8 registers
    so we have 32 bytes
    in parallel we computer s-box for the 32 bytes


"""



def apply_column_mix_on_state_array_element(register_based_data, indecies):
    starting_register_index_1st = indecies[0]
    starting_register_index_2nd = indecies[1]
    starting_register_index_3rd = indecies[2]
    starting_register_index_4th = indecies[3]

    assert starting_register_index_1st != starting_register_index_2nd != starting_register_index_3rd != starting_register_index_4th

    register_pack_1st = tk.get_register_pack(register_based_data, pack_starting_index=starting_register_index_1st)
    register_pack_2nd = tk.get_register_pack(register_based_data, pack_starting_index=starting_register_index_2nd)

    register_pack_1st = xor_two_register_packs(register_pack_1st, register_pack_2nd)
    register_pack_1st = mult_register_pack_by_two(register_pack_1st)
    register_pack_1st = xor_two_register_packs(register_pack_1st, register_pack_2nd)

    register_pack_2nd = tk.get_register_pack(register_based_data, pack_starting_index=starting_register_index_3rd)

    register_pack_1st = xor_two_register_packs(register_pack_1st, register_pack_2nd)

    register_pack_2nd = tk.get_register_pack(register_based_data, pack_starting_index=starting_register_index_4th)

    register_pack_1st = xor_two_register_packs(register_pack_1st, register_pack_2nd)



    return register_pack_1st


def mult_register_pack_by_two(register_pack):
    ret = [[] for _ in range(8)]

    for i in range(32):
        byte = [x[i] for x in register_pack][::-1]
        byte = int(''.join(byte), 2)

        mult_ret = my_galios.mult_by_two_GF_2_8(byte)

        mult_ret = bitstring.BitArray(f'uint8={mult_ret}').bin[::-1]

        for j in range(8):
            ret[j].append(mult_ret[j])

    for i in range(8):
        ret[i] = ''.join(ret[i])

    return ret


def xor_two_register_packs(element_1_registers, element_2_registers):
    pass
    """
    here we have two register packs comping from state_array elements.
    we want to XOR the register couples accordingly

    the result will be a new register pack.

    in verilog we can use a two dimensional array to store registers of each element
    result will also be in such format

    """

    ret = []

    for i in range(8):
        ret.append(tk.xor_two_registers(element_1_registers[i], element_2_registers[i]))

    return ret





def padding(length):
    pass

    """
    padding

    we need to reserve the first two blocks for GCM required values (H and iv0)
    if the data fits in 30 blocks, we pad to that and thus we only need one payload
    otherwise we add payloads of 32 blocks and we pad to that

    results:
    M: append M bits of zero to data
    P: we will need P payloads
    B: how many blocks of data we actually have

    """

    B = int(length / 128)
    Z = 16
    if B * 128 < length:
        Z = int((length - (B * 128 ))/8)

        B = B + 1


    if length < 30 * 128:
        M = 30 * 128 - length
        P = 1


    else:

        L0 = length - (30 * 128)
        P = int ( L0 /  (32 * 128) )

        M = 0

        L1 = L0 - P * 32 * 128

        if (L1 > 0):
            # we have left overs
            P = P + 1
            M = L1

        P = P + 1 # for the first payload

    return [P, M, B, Z]



def get_initial_register_memory(iv, counter, payload_idx = 0):
    global used_counters

    N = 32

    """
    here we generate 32 blocks of data as feed to the cipher
    
    if this is the first 32 blocks (payload_idx = 0):
        the first one, which is all Zeros, represents `H` for GCM.
        the second one, is `iv` plus counter=`nuance`, which is used for the final step of the GMAC
        rest are `iv` plus a counter that starts from (`nuance` + 1)
    else:
        all 32 blocks are made of `iv` plus counter starting from the last seen counter value plus 1
    
    
    note:
        the blocks are stored in revered bit order to respect our register-based memory layout
    """




    block_based_data = []

    for i in range(N):
        used_counters.append(counter)


        # if counter > 2:
        #     counter=2
        # counter = 0 # REMOVE THIS !!!!
        # if counter == 0:
        #     counter = 1
        # else:
        #     counter = 0

        # counter += 1
        # counter = counter % 1

        # print(f'payload_idx: {payload_idx}, i: {i}, counter: {counter}')

        iv_counter = iv + bitstring.BitArray(f"uint32={counter}").bin

        for el in tk.bit_string_to_bit_list(iv_counter)[0][::-1]:
            block_based_data.append(el)

        # counter = tk.gcm_counter(counter)
        counter = counter + 1


    """
    converting the block-based memory to register-based memory
    from now on we work only with the register-based memory
    """
    register_based_data = tk.convert_block_based_to_register_based_memory(block_based_data)
    #
    # recovered_block_based_data = tk.convert_register_based_to_block_based_memory(register_based_data)
    #
    # assert recovered_block_based_data == block_based_data
    #
    # # verification work for the convertor
    # dd = tk.chunks(''.join(block_based_data), 128)
    # verify = []
    # for by in range(16):
    #     for bi in range(8):
    #         t = ''.join([x[by*8 + (7-bi)] for x in dd])
    #         f = tk.chunks(t, 8)
    #         for el in f:
    #             # verify.append(bitstring.BitArray(f'bin={el}').uint8)
    #             verify.append(el)
    #
    #
    # assert verify == register_based_data

    return register_based_data, block_based_data, counter





def apply_round_key(register_based_data, keys_array):
    pass

    """
    APPLY ROUND KEY

    the round key needs to be applied to every register depth 
    (which is the same as one particular block data), once at a time.
    
    if we wanted to work on every resigter depth at once, we'd need to collect the 128 bits
    of data from various locations in the register memory. very wasteful.
    
    instead, we can work one register at a time:
    we can not paralize this work because we are reading and writing on the same memory
    """

    round_key = ''.join(keys_array)

    for register_idx in range(128):
        """
        the key structure needs to match our block structure
        if you just simply read it from left to right, you violate the block structure
        which starts from bit 7 on the most left
        
        you should read the key:
        bit 7  :  0
        bit 15 :  8
        
        """

        idx = (int(register_idx / 8) * 8 ) + (7 - (register_idx - int(register_idx / 8) * 8))

        key_bit = int(round_key[idx])

        if key_bit == 1:

            register = tk.get_register_bus(register_based_data, register_idx)

            updated_register = tk.not_register(register)

            tk.replace_register(register_based_data, register_idx, new_register=updated_register)






def aes_round(register_based_data, keys_array):
    pass

    """
    ok to do in-place
    
    appllying s-box on register-based memory structure:
    
    take one register-pack at a time
    apply s-box on all the 32 bytes (parallel on each register-pack depth)
    """
    
    for i in range(16):
        register_pack_after_sbox = 8 * ['']
        register_pack = tk.get_register_pack(register_based_data, 8 * i)
        for j in range(32):
            byte = ''.join([x[j] for x in register_pack])[::-1]

            s_byte = tk.sbox_rom[int(byte, 2)]


            for bit_idx, bit in enumerate(s_byte[::-1]):
                register_pack_after_sbox[bit_idx] += bit

        tk.replace_register_pack(register_based_data, 8 * i, register_pack_after_sbox)







    """
    can not be done in place
    applying column-mix
    """


    new_rbm =512 * ['']

    for m in range(4):     # m: row
        for n in range(4): # n: col

            # if shit-row was done seperately and before this, we'd use this indecies:
            # mix_column_indexes = [8 * int(4 * n + x) for x in [(m) % 4, (m + 1) % 4, (m + 2) % 4, (m + 3) % 4]]

            # a = mix_col_ind[n][m]
            # b0 = bitstring.BitArray(f"uint4={a[0]}").bin
            # b1 = bitstring.BitArray(f"uint4={a[1]}").bin
            # b2 = bitstring.BitArray(f"uint4={a[2]}").bin
            # b3 = bitstring.BitArray(f"uint4={a[3]}").bin
            #
            # print(f"mix_col_indecies_F13[{n}][{m}] = 16'b{b0}{b1}{b2}{b3};")


            mix_column_indexes = [8*x for x in mix_col_ind[n][m]]

            # print(f'(m, n): ({m}, {n})  {[int(x/8) for x in mix_column_indexes]}')

            ret_register_pack = apply_column_mix_on_state_array_element(register_based_data, indecies = mix_column_indexes)

            pack_starting_index = 8 * int(4 * n + (m % 4))

            tk.replace_register_pack(new_rbm, pack_starting_index, ret_register_pack)


    register_based_data = copy.deepcopy(new_rbm)





    """
    applying the round key
    """

    apply_round_key(register_based_data, keys_array)

    return register_based_data





def generate_gcm_prexor_cipher_texts(register_based_data, keys_arrays):
    pass

    """
    apply the first key before initiating the rounds
    """


    apply_round_key(register_based_data, keys_arrays[0])


    """
    applying the rounds
    """
    for i in range(14):
        # print(f'\nround {i}')
        register_based_data = aes_round(register_based_data, keys_arrays[i + 1])


    prexor_cipher_texts = tk.convert_register_based_to_block_based_memory(register_based_data)

    # print(tk.chunks(prexor_cipher_texts, 16))


    return prexor_cipher_texts




def aes_256_gcm_encrypt(data_bin, auth_bin, keys_arrays, iv, mode, get_auth_tag=True):
    global used_counters

    payload_counts, padding_bits_count, blocks_count, last_block_relevant_bytes = padding(len(data_bin))

    data_bin = data_bin + padding_bits_count * '0'
    data_bin_blocks = tk.chunks(data_bin, 128)

    px_cipher_text_payloads = []

    counter_G12 = 0

    for i in range(payload_counts):
        print(f'encrypting payload {i} of {payload_counts}')
        register_based_data, _ , counter_G12 = get_initial_register_memory(iv, counter_G12, i)

        px_cipher_text_payload = generate_gcm_prexor_cipher_texts(register_based_data, keys_arrays)

        px_cipher_text_payloads.append(px_cipher_text_payload)

    # print('\npx_cipher_text_payloads:')
    # for el in tk.chunks(px_cipher_text_payloads[0], 16):
    #     print(tk.bin_list_to_hex_string(el))

    """
    Terminology:
    
    px_cipher_text:
        a 128-bit long encrypted bitstring generated by encrypting one block
        
    px_cipher_text_payload:
        a set of 32 px_cipher_text generated by parallel encryption of a payload
        
    payload:
        a set of 32 blocks of data (here data is `iv` plus a counter)
        
    cipher_text:
        a 128-bit long bitstring resulted from XORing px_cipher_text to its relevant input data_block
        
        
    
    at this point we have all the payload_px_cipher texts in block_format.
    now we XOR them with the provided input.
    the input is plain text if we are encrypting.
    the input is cipher text if we are decrypting.
    
    every payload_px_cipher text contains 32 block cipher text
    the first two px_cipher_text of the first px_cipher_text_payload is reserved for GCM.
    so we start from the third px_cipher_text of the first px_cipher_text_payload
    
    
    we go through all the px_cipher_text_payload
    and collect as many as `blocks_count` cases of px_cipher_text from them
    we xor those with our input data to generate the cipher_text
    """

    cipher_texts = []

    blocks_counter = 0

    for counter_G11 in range(payload_counts):


        if counter_G11 == 0:
            counter_G12 = 2
        else:
            counter_G12 = 0

        # print("")
        while counter_G12 < 32:


            # print(f'XORing {blocks_counter}th of data_bin_blocks with {counter_G12}th block of the {counter_G11}th px_cipher_text_payload to generate {blocks_counter}th cipher text')

            left_xor = data_bin_blocks[blocks_counter]
            right_xor= tk.get_block_bus(px_cipher_text_payloads[counter_G11], counter_G12)



            cipher_texts.append(tk.xor_two_registers(left_xor, right_xor))

            blocks_counter += 1

            if blocks_counter == blocks_count:
                break

            counter_G12 += 1



    # zero the redundant bytes of the last cipher text block

    cipher_texts[-1] =cipher_texts[-1][:last_block_relevant_bytes * 8] + 8 * (16 - last_block_relevant_bytes) * '0'

    if not get_auth_tag:
        return cipher_texts
    """
    building auth_tag after the cipher texts are at hand
    tk.chunks(''.join(cipher_texts), 8)
    """

    H                       = tk.bit_string_to_bit_list(tk.get_block_bus(px_cipher_text_payloads[0], 0))[0]
    iv_counter_0_enctypted  = tk.bit_string_to_bit_list(tk.get_block_bus(px_cipher_text_payloads[0], 1))[0]


    auth_bin = tk.zero_pad_bin_list_to_multiples_of_128(auth_bin)[0]


    auth_tag = my_galios.H_mult(auth_bin, H)

    # print(f"\nauth_tag_after_first_mult = {tk.bin_list_to_hex_string(auth_tag)}")

    # print(f"gcm_H_prexor_G: {tk.bin_list_to_hex_string(H)}")
    # print(f"gcm_auth_bin_G: {tk.bin_list_to_hex_string(auth_bin)}")
    # print(f"iv_counter_0_enctypted_G: {tk.bin_list_to_hex_string(iv_counter_0_enctypted)}")
    # print(f"auth_tag: {tk.bin_list_to_hex_string(auth_tag)}")

    # print("\nstarting the loop")

    # applying cipher texts
    for i in range(blocks_count):
        if mode == 'encrypt':
            auth_tag = tk.xor_two_arrays(auth_tag, tk.bit_string_to_bit_list(cipher_texts[i])[0])
        else:
            auth_tag = tk.xor_two_arrays(auth_tag, tk.bit_string_to_bit_list(data_bin_blocks[i])[0])

        # print(f"auth_tag_{i+1} after xor  = {tk.bin_list_to_hex_string(auth_tag)}")

        auth_tag = my_galios.H_mult(auth_tag, H)
        # print(f"auth_tag_{i+1} after mult = {tk.bin_list_to_hex_string(auth_tag)}")

        dgf=5454
    # print("end of the loop\n")
    # print(f"auth_tag befor lens = {tk.bin_list_to_hex_string(auth_tag)}")


    # applying lengths




    len_A = 128 # auth bin is always 16 bytes
    len_C = (blocks_count-1) * 128 + last_block_relevant_bytes * 8

    cipher_texts = ''.join(cipher_texts)[:len_C]


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





    # assert len(set(used_counters)) == len(used_counters)

    used_counters = []



    return cipher_texts, auth_tag



def main_efficient():
    pass

    with open("aes_256_key.txt", 'r') as my_file:
        key_hex =  my_file.read()

    with open("aes_256_data.txt", 'r') as my_file:
        data_hex =  my_file.read()

    with open("aes_256_data.rgb", 'r') as my_file:
        data_rgb = my_file.read().split('\n')[:-1]


    # iv = ''.join(np.random.choice(['0', '1'], 96))
    iv = tk.hex_string_to_bit_string('cafebabefacedbaddecaf888')
    # iv = tk.hex_string_to_bit_string('000000000000000000000000')

    assert len(iv) == 96

    auth_hex= "90e87315fb7d4e1b4092ec0cbfda5d7d"

    # data_bin = tk.hex_string_to_bit_string(data_hex)
    data_bin = ''.join([bitstring.BitArray(f"uint16={x}").bin for x in data_rgb])
    key_bin = tk.hex_string_to_binary_list(key_hex)
    auth_bin = tk.hex_string_to_binary_list(auth_hex)

    # write input memory with plain text as input (for encryption)

    keys_arrays, keys_arrays_hex =  aes_key_expansion_256.expand_keys(key_bin)


    cipher_texts, auth_tag_encrypt = aes_256_gcm_encrypt(data_bin, auth_bin, keys_arrays, iv, mode='encrypt')


    # save cipher_text as 16-bit pixels into a file:
    with open('cipher_pixels.txt', 'w') as my_file:
        for el in tk.cipher_text_to_k_bit_pixels(cipher_texts, 16):
            my_file.write(f'{el}\n')

    with open('cipher_text.txt', 'w') as my_file:
        my_file.write(cipher_texts)



    # exit()

    # write input memory with cipher text as input (for decryption)


    assert len(cipher_texts) == len(data_bin)




    # manupulating the cipher text
    # a=copy.deepcopy(cipher_texts)
    # cipher_texts  = tk.randomly_manupulate_one_bit_in_bin_list(cipher_texts)
    # assert not a==cipher_texts




    plain_texts, auth_tag_decrypt = aes_256_gcm_encrypt(cipher_texts, auth_bin, keys_arrays, iv, mode='decrypt')

    plain_texts = ''.join(plain_texts)

    assert auth_tag_encrypt == auth_tag_decrypt, f'{tk.bin_list_to_hex_string(auth_tag_encrypt)} , {tk.bin_list_to_hex_string(auth_tag_decrypt)}'

    assert len(plain_texts) >= len(data_bin)


    for i in range(len(data_bin)):
        assert plain_texts[i] == data_bin[i], i


    assert plain_texts[:len(data_bin)] == data_bin


    print('\n cipher texts:')
    for el in tk.chunks(cipher_texts, 128):
        print(tk.bits_to_hex_string(el))

    sdf=4
    # for payload_idx, pa



if __name__ == '__main__':
    main_efficient()