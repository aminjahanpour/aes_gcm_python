import bitstring
from PIL import Image, ImageDraw
from math import acos, sqrt
import numpy as np
import aes_toolkit as tk
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import cv2
from statsmodels.stats.weightstats import DescrStatsW

import aes_key_expansion_256
from aes_encoder_efficient import aes_256_gcm_encrypt
import numpy as np
import requests
import json
import math
from scipy.stats import entropy

key_missing_bits_count = 7



"""
guess a key
decrypt the cipher text
calculate max of hue_hist
what is the hue entropy of the frame?
it is faithful to key guess quality?
 

the first few pixels are enough for an evaluation of cost function.
hue of neighbouring cipher pixels, as opposed to a regular image, changes so very drastically.
one 128-bit block of 8 pixels would suffice for evaluation.

"""
def get_hue_hist_from_pil_image(pil_image):

    open_cv_image_cipher_text = np.array(pil_image)

    img_bgr_cipher_text = open_cv_image_cipher_text[:, :, ::-1].copy() # Convert RGB to BGR

    img_hsv = cv2.cvtColor(img_bgr_cipher_text, cv2.COLOR_BGR2HSV)
    hue_hist_full_frame = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])


    # hue_var = DescrStatsW(list(range(180)), weights=[x[0] for x in hue_hist_full_frame], ddof=0).std

    hue_entropy = entropy([x[0]/256 for x in hue_hist_full_frame],base=2)

    hue_max = max([x[0] for x in hue_hist_full_frame])

    return hue_hist_full_frame, hue_entropy, hue_max



def evaluate_cipher_image(cipher_pixels, draw_plot=True):

    # read the cipher text






    width = 16
    height = 16

    img_rgb_cipher_test = Image.new('RGB', (width,height))

    counter = 0


    for mm in range(height):

        for nn in range(width):

            cipher_pixel = int(cipher_pixels[counter])

            r_rec = (cipher_pixel >> 11 & 31) << 3
            g_rec = (cipher_pixel >> 5 & 63) << 2
            b_rec = (cipher_pixel & 31) << 3

            img_rgb_cipher_test.putpixel((nn, mm), (r_rec, g_rec, b_rec))



            counter += 1

    cipher_hue_hist, hue_entropy, hue_max = get_hue_hist_from_pil_image(img_rgb_cipher_test)

    if draw_plot:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        ax1.imshow(img_rgb_cipher_test)



        ax2.plot(cipher_hue_hist, label=f'decrypted, var:{round(hue_entropy, 4)}')



        # original image
        pil_image = Image.open('C:\\Users\\jahan\Desktop\\verilog\\deflate\\AEB_python\\shuttle_16.bmp').convert('RGB')
        org_hue_hist, org_hues_var, _ = get_hue_hist_from_pil_image(pil_image)
        ax2.plot(org_hue_hist, label=f'org_hue_hist, var:{round(org_hues_var, 4)}')

        plt.legend()
        plt.show()

    return hue_entropy, hue_max

global opt_settings
opt_settings = {}



def obj_fun(individual):
    global opt_settings




    key_missing_bits_guess = individual


    # apply the key guess and run the decryption
    key_bin = tk.hex_string_to_bit_string(opt_settings['key_hex'])

    key_bin = key_bin[:-opt_settings['dim']] + key_missing_bits_guess

    assert len(key_bin) == 256

    key_bin = tk.chunks(key_bin, 8)

    keys_arrays, keys_arrays_hex =  aes_key_expansion_256.expand_keys(key_bin)

    data_bin = opt_settings['cipher_text']

    iv = tk.hex_string_to_bit_string('cafebabefacedbaddecaf888')


    cipher_texts = aes_256_gcm_encrypt(data_bin, '', keys_arrays, iv, mode='decrypt', get_auth_tag=False)

    cipher_texts = ''.join(cipher_texts)

    cost_function_1, cost_function_2 = evaluate_cipher_image(tk.cipher_text_to_k_bit_pixels(cipher_texts, 16), draw_plot=False)


    with open('cost_function.log', 'a') as my_file:
        my_file.write(f'{key_missing_bits_guess}, {cost_function_1}, {cost_function_2}\n')


    return np.array([-cost_function_1, -cost_function_2])


def key_search():

    global opt_settings

    # read the cipher text

    with open("cipher_text.txt", 'r') as my_file:
        cipher_text = my_file.read()

    with open("aes_256_key.txt", 'r') as my_file:
        key_hex =  my_file.read()



    dim = key_missing_bits_count


    opt_settings = {
        'cipher_text': cipher_text,
        'key_hex'    : key_hex,
        'dim'        : dim
    }


    # brute force
    for i in range(2 ** key_missing_bits_count):
        dv = bitstring.BitArray(f"uint{key_missing_bits_count}={i}").bin
        print(f'i: {i} from {2 ** key_missing_bits_count}, key: {dv}')
        ret = obj_fun(dv)





    server = 'http://127.0.0.1:5000/'
    key = '0d80c5a7740ac8ff2fc29dc4a5d791b400161b21'
    budget = int(100)
    id = 'dd'
    piac = 0
    initial = ''
    num_objs = 2
    binary_problem = 1

    resp = requests.post(url='%s?key=%s&req=del&id=%s' % (server, key, id))
    print(resp.content)

    resp = requests.post(url='%s?key=%s&req=create&id=%s&dim=%s&budget=%s&piac=%s&initial=%s&num_objs=%s&binary_problem=%s' % (server, key, id, dim, budget, piac, initial, num_objs, binary_problem))

    print(resp.content)

    resp = requests.post(url='%s?key=%s&req=ask&id=%s' % (server, key, id))
    print(resp.content)


    f_best = 10e20
    counter = 0
    while b'budget_used_up' not in resp.content:
        dv = json.loads(resp.content.decode("utf-8"))["dv"]

        # creating the vector of objective functions from recoeved solutions
        f = np.array(list(map(obj_fun, dv)))

        if num_objs == 1:
            # updating the best objective function (for logging only)
            if min(f) <= f_best: #update f_best and x_best
                f_best = min(f)
                x_best = dv[list(f).index(min(f))]

            f_arrstr = np.char.mod('[%f]', f)
            f_string = ";".join(f_arrstr)         # '[1.2];[1.3]'
            print("progress: %s   f_best:%f, best x:%s" % (round(100. * counter / budget, 0), f_best, dvs_to_key_string(x_best)))

        else:
            f_arrstr = [str(list(x)) for x in f]
            f_string = ";".join(f_arrstr)        # '[1.2,6.5];[1.3, 1.2]'
            print("progress: %s " % (round(100. * counter / budget, 2)))

        payload = {'req': 'roll',
                   'key': key,
                   'id': id,
                   'dim': dim,
                   'f': f_string
                   }

        resp = requests.post(url=server, json=payload)
        # print(resp.content)

        counter += len(dv)


    resp = requests.post(url='%s?key=%s&req=results&id=%s' % (server, key, id))
    print(resp.content)




    f=json.loads(resp.content.decode("utf-8"))["best_f"]
    best_dv=json.loads(resp.content.decode("utf-8"))["best_dv"]
    for el in best_dv:
        print(''.join([str(int(x)) for x in el]))
    import matplotlib.pyplot as plt
    plt.scatter(x=[x[0] for x in f], y=[x[1] for x in f])
    plt.show()

    # print(f'i: {i}, key_missing_bits_guess:{key_missing_bits_guess}, cost_function: {cost_function}')
    # sdf=5


if __name__ == '__main__':
    key_search()

    # with open("cipher_pixels.txt", 'r') as my_file:
    #     cipher_pixels = my_file.read().split('\n')[:-1]
    # #
    # evaluate_cipher_image(cipher_pixels)
