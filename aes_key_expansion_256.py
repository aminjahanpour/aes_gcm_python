import bitstring

import aes_toolkit as tk


verbose = 0

def rot_word(in_vector):

    out_vector = len(in_vector) * [0]

    out_vector[0] = in_vector[1]
    out_vector[1] = in_vector[2]
    out_vector[2] = in_vector[3]
    out_vector[3] = in_vector[0]

    if verbose: print(f"D2: before_rot_wor:{in_vector}, after_rot_word_D: {out_vector}")

    return out_vector

def apply_round_constant(in_vector, idx):

    b = bitstring.BitArray(f"uint8={tk.round_constants[idx]}").bin

    bb = [b, 8*'0', 8*'0', 8*'0']

    ret=tk.xor_two_arrays(in_vector, bb)

    if verbose: print(f"D1: rc in:         {in_vector}");

    if verbose: print(f"D1: rc out:        {ret}");
    return ret





def get_all_keys(key_array):

    base_key_words = [key_array[i:i + 4] for i in range(0, len(key_array), 4)]

    # words_readable = [print_key_column(el) for el in base_key_words]

    idx = 0

    words = []
    words_readable = []

    eight_counter = 0

    new_word = base_key_words[-1]

    while True:


        if eight_counter == 0:
            stage = 'new'

        elif eight_counter == 4:
            stage = 'middle'

        else:
            stage = 'regular'



        if stage == 'regular':
            left_xor = new_word
            right_xor = base_key_words[eight_counter]


        elif stage == 'middle':
            left_xor = tk.get_sub_bytes(new_word)
            right_xor = base_key_words[eight_counter]


        elif stage == 'new':
            left_xor = rot_word(new_word)

            left_xor = tk.get_sub_bytes(left_xor)

            left_xor = apply_round_constant(left_xor, idx)
            idx += 1

            if len(words) >= 8:
                if verbose: print('\nswitching base_key_words')
                base_key_words = words[-8:]

            if verbose: print(f'base_key_words_D[0]: {base_key_words[0]}')
            if verbose: print(f'base_key_words_D[1]: {base_key_words[1]}')
            if verbose: print(f'base_key_words_D[2]: {base_key_words[2]}')

            eight_counter = 0

            right_xor = base_key_words[eight_counter]


        # print_key_column(left_xor), print_key_column(right_xor), print_key_column(new_word)
        new_word = tk.xor_two_arrays(left_xor, right_xor)

        words.append(new_word)
        words_readable.append(tk.print_key_column(new_word))


        if len(words) == 52:
            break

        eight_counter += 1

        if eight_counter == 8:
            eight_counter = 0



    return words, words_readable





def collect_row_keys_to_rows(all_keys_columned,all_keys_columned_hex):

    keys_hex = []
    keys = []

    for key_counter in range(15):
        key_hex = []
        key = []

        for j in range(key_counter * 4, (key_counter+1) * 4):

            key_hex += all_keys_columned_hex[j]
            [key.append(el) for el in all_keys_columned[j]]


        key_hex = "".join(key_hex)
        keys_hex.append(key_hex)

        keys.append(key)

    return keys, keys_hex




def expand_keys(base_key_bitstring):

    base_key_bitstring_hex = tk.bin_list_to_hex_string(base_key_bitstring)

    key_1_hex = base_key_bitstring_hex[:32]
    key_2_hex = base_key_bitstring_hex[32:]


    key_1 = base_key_bitstring[:16]
    key_1 = [key_1[i:i + 4] for i in range(0, len(key_1), 4)]

    key_2 = base_key_bitstring[16:]
    key_2 = [key_2[i:i + 4] for i in range(0, len(key_2), 4)]



    key = key_1_hex + key_2_hex


    base_key = tk.hex_string_to_binary_list(key)

    words, words_readable = get_all_keys(base_key)


    key_1_hex = [key_1_hex[i:i + 8] for i in range(0, len(key_1_hex), 8)]
    key_2_hex = [key_2_hex[i:i + 8] for i in range(0, len(key_2_hex), 8)]

    keys, keys_hex = collect_row_keys_to_rows(key_1 + key_2 + words, key_1_hex + key_2_hex + words_readable)

    return [keys, keys_hex]




if __name__ == '__main__':

    my_file = open('C:\\Users\\jahan\\Desktop\\verilog\\crypto\\AES_python\\aes_256_key.txt', 'r')
    base_key_hex_string = my_file.read()
    my_file.close()

    # base_key_hex_string="603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"

    base_key_bitstring= tk.hex_string_to_binary_list(base_key_hex_string)

    keys, keys_hex = expand_keys(base_key_bitstring)


    # assert keys == [['01100000', '00111101', '11101011', '00010000', '00010101', '11001010', '01110001', '10111110', '00101011', '01110011', '10101110', '11110000', '10000101', '01111101', '01110111', '10000001'], ['00011111', '00110101', '00101100', '00000111', '00111011', '01100001', '00001000', '11010111', '00101101', '10011000', '00010000', '10100011', '00001001', '00010100', '11011111', '11110100'], ['10011011', '10100011', '01010100', '00010001', '10001110', '01101001', '00100101', '10101111', '10100101', '00011010', '10001011', '01011111', '00100000', '01100111', '11111100', '11011110'], ['10101000', '10110000', '10011100', '00011010', '10010011', '11010001', '10010100', '11001101', '10111110', '01001001', '10000100', '01101110', '10110111', '01011101', '01011011', '10011010'], ['11010101', '10011010', '11101100', '10111000', '01011011', '11110011', '11001001', '00010111', '11111110', '11101001', '01000010', '01001000', '11011110', '10001110', '10111110', '10010110'], ['10110101', '10101001', '00110010', '10001010', '00100110', '01111000', '10100110', '01000111', '10011000', '00110001', '00100010', '00101001', '00101111', '01101100', '01111001', '10110011'], ['10000001', '00101100', '10000001', '10101101', '11011010', '11011111', '01001000', '10111010', '00100100', '00110110', '00001010', '11110010', '11111010', '10111000', '10110100', '01100100'], ['10011000', '11000101', '10111111', '11001001', '10111110', '10111101', '00011001', '10001110', '00100110', '10001100', '00111011', '10100111', '00001001', '11100000', '01000010', '00010100'], ['01101000', '00000000', '01111011', '10101100', '10110010', '11011111', '00110011', '00010110', '10010110', '11101001', '00111001', '11100100', '01101100', '01010001', '10001101', '10000000'], ['11001000', '00010100', '11100010', '00000100', '01110110', '10101001', '11111011', '10001010', '01010000', '00100101', '11000000', '00101101', '01011001', '11000101', '10000010', '00111001'], ['11011110', '00010011', '01101001', '01100111', '01101100', '11001100', '01011010', '01110001', '11111010', '00100101', '01100011', '10010101', '10010110', '01110100', '11101110', '00010101'], ['01011000', '10000110', '11001010', '01011101', '00101110', '00101111', '00110001', '11010111', '01111110', '00001010', '11110001', '11111010', '00100111', '11001111', '01110011', '11000011'], ['01110100', '10011100', '01000111', '10101011', '00011000', '01010000', '00011101', '11011010', '11100010', '01110101', '01111110', '01001111', '01110100', '00000001', '10010000', '01011010'], ['11001010', '11111010', '10101010', '11100011', '11100100', '11010101', '10011011', '00110100', '10011010', '11011111', '01101010', '11001110', '10111101', '00010000', '00011001', '00001101'], ['11111110', '01001000', '10010000', '11010001', '11100110', '00011000', '10001101', '00001011', '00000100', '01101101', '11110011', '01000100', '01110000', '01101100', '01100011', '00011110']]
    # assert keys_hex == ['603deb1015ca71be2b73aef0857d7781', '1f352c073b6108d72d9810a30914dff4', '9ba354118e6925afa51a8b5f2067fcde', 'a8b09c1a93d194cdbe49846eb75d5b9a', 'd59aecb85bf3c917fee94248de8ebe96', 'b5a9328a2678a647983122292f6c79b3', '812c81addadf48ba24360af2fab8b464', '98c5bfc9bebd198e268c3ba709e04214', '68007bacb2df331696e939e46c518d80', 'c814e20476a9fb8a5025c02d59c58239', 'de1369676ccc5a71fa2563959674ee15', '5886ca5d2e2f31d77e0af1fa27cf73c3', '749c47ab18501ddae2757e4f7401905a', 'cafaaae3e4d59b349adf6acebd10190d', 'fe4890d1e6188d0b046df344706c631e']

    print(keys)
    [print(key_hex) for key_hex in keys_hex]


