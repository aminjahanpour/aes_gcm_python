# AES GCM in Python

This is pure python code for an AES GCM cipher.

There are two implementations in this repo.

#### Conventional implementation
It follows the classic prosses of AES.
It is great for learning.

#### Efficient implementation
This implementation is based on the work of [this paper](fast_aes.pdf).
It uses bit-slicing techniques to increase the throughput of the cipher.

The AES here works in GCM operating mode.

`python aes_encoder_efficient.py
`

A sad attempt to break the cipher is archived in [cipher_analysis.py](cipher_analysis.py) where I try
to decrypt an image encrypted with the cipher. You could get something using ESB operating mode ([see this](cipher_text_rgb.bmp)) but
absolutely nothing with the GCM.

