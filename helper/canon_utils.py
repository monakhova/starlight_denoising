import numpy as np

def read_16bit_raw(filename, height = 1280, width = 2160):
    pgmf = open(filename, 'rb')

    raster = []
    for y in range(height):
        row = []
        for yy in range(width):
            low_bits = ord(pgmf.read(1))
            row.append(low_bits+255*ord(pgmf.read(1)))
        raster.append(row)

    pgmf.close()
    return np.asarray(raster)

def raw_to_4(im):
    native_1 = im[0::2,0::2] # top left (R)
    native_2 = im[1::2,0::2] # bottom left (IR)
    native_3 = im[0::2,1::2] # top right (G)
    native_4 = im[1::2,1::2] # botom right (B)

    col_list = [native_1,  native_3, native_4, native_2]
    return np.asarray(col_list).transpose(1,2,0)