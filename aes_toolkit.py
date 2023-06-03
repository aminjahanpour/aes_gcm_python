import bitstring
import numpy as np
import copy

s_box = [
            [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76],
            [0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0],
            [0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15],
            [0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75],
            [0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84],
            [0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF],
            [0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8],
            [0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2],
            [0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73],
            [0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB],
            [0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79],
            [0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08],
            [0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A],
            [0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E],
            [0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF],
            [0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
            ]


round_constants = [1, 2, 4, 8, 16, 32, 64]


global sbox_rom
sbox_rom = []


"""
key length      128     192     256
rounds count    10      12      14


"""

aes_versions = {
    'AES_128': {'rounds_count': 10, 'key_length': 128},
    'AES_256': {'rounds_count': 14, 'key_length': 256},
}

# version = 'AES_128'
version = 'AES_256'

conf = aes_versions[version]


def chunks(input, n):
    return [input[i:i + n] for i in range(0, len(input), n)]


def build_sbox_rom():
    global sbox_rom

    # populated such that the index is: left_nibble + right_nible
    for idx in range(16 * 16):
        idx_bits = bitstring.BitArray(f"uint8={idx}").bin
        left_nible = idx_bits[:4]
        right_nible = idx_bits[4:]
        value = s_box[int(left_nible, 2)][int(right_nible, 2)]
        sbox_rom.append(bitstring.BitArray(f"uint8={value}").bin)

build_sbox_rom()

def xor_two_arrays(m,n):

    final = []

    for x,y in zip(m,n):

        ret = ''

        for x1, y1 in zip(x,y):
            ret += str(int(x1)^int(y1))

        final.append(ret)

    return final


def xor_two_registers(a, b):
    l = len(a)
    m = bitstring.BitArray(f'bin={a}').uint
    n = bitstring.BitArray(f'bin={b}').uint
    ans =  m ^ n

    return bitstring.BitArray(f'uint{l}={ans}').bin

def get_sub_bytes(in_array):
    global sbox_rom

    ret = []

    for el in in_array:
        ret.append(sbox_rom[int(el, 2)])


    # print(f'before_sub_byte_D: {in_array}, after_sub_byte_D: {ret}')

    return ret



"""
convertors
"""

def bin_list_to_hex_string(input):
    ret = ''
    try:
        for el in input:
            for by in el:
                ret += bitstring.BitArray(f"bin={by}").hex

        return ret
    except:
        for el in input:
            ret += bitstring.BitArray(f"bin={el}").hex

        return ret
def bits_to_hex_string(input):
    ret = ''
    for el in [input[i:i + 8] for i in range(0, len(input), 8)]:
        ret += bitstring.BitArray(f"bin={el}").hex

    return ret

def ascii_string_to_hex_string(input):
    ret = ''
    for el in input:
        ret += bitstring.BitArray(f"uint8={ord(el)}").hex

    return ret

def hex_string_to_binary_list(input):
    chunks_hex = [input[i:i + 2] for i in range(0, len(input), 2)]
    ret = []
    for chunk in chunks_hex:
        ret.append(bitstring.BitArray(f"hex={chunk}").bin)

    return ret

def hex_string_to_bit_string(input):
    ret = ''
    chunks_hex = [input[i:i + 2] for i in range(0, len(input), 2)]
    for chunk in chunks_hex:
        ret += bitstring.BitArray(f"hex={chunk}").bin

    return ret


def bin_list_to_int_list(input):
    ret = []

    for el in input:
        ret.append(bitstring.BitArray(f"bin={el}").uint)

    return ret


def bit_string_to_bit_list(input):

    input_bytes = [input[i:i + 8] for i in range(0, len(input), 8)]

    return [input_bytes[i:i + 16] for i in range(0, len(input_bytes), 16)]


def bit_list_to_bit_string(input):
    ret = ''
    for el in input:
        ret += ''.join(el)

    return ret

def int_list_to_bin_list(input):
    ret = []
    for el in input:
        ret.append(bitstring.BitArray(f"uint8={el}").bin)

    return ret


def bin_list_to_poly(input):
    return [int(x) for x in "".join(input)]


def bin_list_to_integer(input):
    return bitstring.BitArray(f"bin={''.join(input)}").uint

def int_to_bin_list(input):

    assert input <= 2**128

    bin = bitstring.BitArray(f"uint128={input}").bin
    return [bin[i:i + 8] for i in range(0, len(bin), 8)]






"""
printers
"""
def print_key_column(col):
    col="".join(col)

    ret = ''
    chunks = [col[i:i + 8] for i in range(0, len(col), 8)]

    for chunk in chunks:
        ret += bitstring.BitArray(f"bin={chunk}").hex

    return ret


def display_array_in_hex(in_array):
    hexes = []
    for el in in_array:
        hexes.append(bitstring.BitArray(f"uint8={int(el,2)}").hex)

    ret = [
        [hexes[0], hexes[1], hexes[2], hexes[3]],
        [hexes[4], hexes[5], hexes[6], hexes[7]],
        [hexes[8], hexes[9], hexes[10], hexes[11]],
        [hexes[12], hexes[13], hexes[14], hexes[15]],
    ]

    return ret


def print_keys_hex_array(input):
    a = "".join(input)
    chunks_hex = [a[i:i + 8] for i in range(0, len(a), 8)]
    for chunk in chunks_hex:
        print(chunk)

def display_vector_in_hex(in_vector):
    hexes = []
    for el in in_vector:
        hexes.append(bitstring.BitArray(f"uint8={int(el,2)}").hex)

    ret = [
        [hexes[0]],
        [hexes[1]],
        [hexes[2]],
        [hexes[3]],
    ]

    return ret


def display_register_based_memory(register_based_data):

    for i in range(128):
        print(f'\nregister: {i}')
        for j in range(4):
            idx = i * 4 + j
            print(f"byte[{j}] =  {register_based_data[idx]}")






def randomly_manupulate_one_bit_in_bin_list(input):
    bit_idx = int(len(input) * np.random.random())

    bit = input[bit_idx]

    if bit=='0':
        sub = '1'
    elif bit == '1':
        sub = '0'
    else:
        raise

    return input[:bit_idx] + sub + input[bit_idx+1:]












def convert_block_based_to_register_based_memory(in_arr):
    out_arr = 128 * 4 * [0]

    counter_16 = 0
    block_counter = 0
    aux_counter = 0


    for el in in_arr:

        involved_registers = [counter_16 * 8 + x for x in range(8)]

        bit_counter = 7

        for involved_register in involved_registers:

            target_row_idx_in_register_mem = 4 * involved_register + int(block_counter / 8)

            bit_value = int(el[bit_counter])

            out_arr[target_row_idx_in_register_mem] = (out_arr[target_row_idx_in_register_mem] << 1) | bit_value

            bit_counter -= 1

        aux_counter += 1
        counter_16 += 1

        if aux_counter == 16:
            aux_counter = 0
            block_counter += 1

        if counter_16 == 16:
            counter_16 = 0


    return [bitstring.BitArray(f'uint8={x}').bin for x in out_arr]



def convert_register_based_to_block_based_memory(in_arr):

    out_arr = 32 * 16 * [0]

    four_counter = 0
    register_counter = 0


    for byte_counter in range(4 * 128):

        register_byte = in_arr[byte_counter]

        involved_blocks = [four_counter * 8 + x for x in range(8)]

        bit_counter = 0

        for involved_block in involved_blocks:
            bit = int(register_byte[bit_counter])

            target_addr = involved_block * 16 + int(register_counter / 8)

            out_arr[target_addr] = out_arr[target_addr] | (bit << (register_counter % 8))



            bit_counter += 1



        four_counter += 1
        if four_counter == 4:
            four_counter = 0
            register_counter += 1

    return [bitstring.BitArray(f'uint8={x}').bin for x in out_arr]



def apply_round_key_to_block_based_memory(block_based_memory, key):

    ret = []

    for idx, block_bytes in enumerate(chunks(block_based_memory, 16)):
        a = xor_two_arrays(block_bytes, key)
        for el in a:
            ret.append(el)

    return ret


def get_register_bus(register_based_data, register_idx):
    ret = ''

    for i in range(4):
        ret += register_based_data[4 * register_idx + i]

    return ret


def get_register_pack(register_based_data, pack_starting_index):
    return [get_register_bus(register_based_data, pack_starting_index + x) for x in range(8)]


def not_register(register):
    ret = ''
    for el in register:
        if el=='0':
            ret+='1'
        else:
            ret+='0'

    return ret


def replace_register(register_based_data, register_idx, new_register):
    counter = 0
    for byte in chunks(copy.deepcopy(new_register), 8):
        target_row_idx = register_idx * 4 + counter

        register_based_data[target_row_idx] = byte

        counter += 1



def replace_register_pack(register_based_data, pack_starting_index, new_register_pack):
    for i in range(8):
        register_idx = pack_starting_index + i

        updated_register = new_register_pack[i]

        replace_register(register_based_data=register_based_data,
                            register_idx=register_idx,
                            new_register=updated_register)


def get_block_bus(block_based_data, block_idx):
    ret = ''

    for i in range(16):
        ret += block_based_data[16 * block_idx + i]

    return ret


def compare_two_register_sets(key,set_1, set_2):
    for i in range(128):
        r1 = get_register_bus(set_1, i)
        r2 = get_register_bus(set_2, i)
        # print(f'{i}, key: {bit_list_to_bit_string(key)[i]}, org reg: {r1}, reg: {r2},    {r1 == r2}')
        assert bool(int(bit_list_to_bit_string(key)[i])) == (not (r1 == r2))


def compare_two_blocks(key, set_1, set_2, idx):
    b1 = get_block_bus(set_1, idx)
    b2 = get_block_bus(set_2, idx)
    key = bit_list_to_bit_string(key)

    print(f'key: {bit_string_to_bit_list(key)}')
    print(f'b1 : {bit_string_to_bit_list(b1)}')
    print(f'b2 : {bit_string_to_bit_list(b2)}')
    for i in range(128):
        print(f'{i}, key: {key[i]}, block_1: {b1[i]}, block_2: {b2[i]},    {b1[i] == b2[i]}')
        # assert bool(int(bit_list_to_bit_string(key)[i])) ==( not (b1[i] == b2[i]))










def zero_pad_bin_list_to_multiples_of_128(input):

    if input == []:
        data_chunks = [16 * ['00000000']]
    else:
        data_chunks = [input[i:i + 16] for i in range(0, len(input), 16)]

        if len(data_chunks[-1]) < 16:
            data_chunks[-1] = data_chunks[-1] + (16 - len(data_chunks[-1])) * ['00000000']

    return data_chunks



def gcm_counter(input):

    if(input == 0):
        ret = (311 * 193939) % (2 ** 32)
    else:
        ret = (input * 311) % (2 ** 32)

    return ret
















def cipher_text_to_k_bit_pixels(cipher_text, k):
    return [bitstring.BitArray(f"bin={el}").uint for el in chunks(cipher_text, k)]

