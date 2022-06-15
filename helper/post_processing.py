import numpy as np
import cv2
import scipy.io

def bayer_bilinear(imagei, height = None, width= None):
    
    if len(imagei.shape) == 4:
        frames, img_height, img_width, channels = imagei.shape
    elif len(imagei.shape) == 3:
        img_height, img_width, channels = imagei.shape
        frames = 1
        imagei = imagei[np.newaxis]
    else:
        print('invalid image size')
        
    if height is None or width is None:
        height = img_height*2
        width = img_width*2

    new_image = np.empty((frames, height, width, channels))
    for k in range(0,frames):
        for i in range(0,channels):
            image = imagei[k,...,i].ravel()

            x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
            y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

            y, x = np.divmod(np.arange(height * width), width)

            x_l = np.floor(x_ratio * x).astype('int32')
            y_l = np.floor(y_ratio * y).astype('int32')

            x_h = np.ceil(x_ratio * x).astype('int32')
            y_h = np.ceil(y_ratio * y).astype('int32')

            x_weight = (x_ratio * x) - x_l
            y_weight = (y_ratio * y) - y_l

            a = image[y_l * img_width + x_l]
            b = image[y_l * img_width + x_h]
            c = image[y_h * img_width + x_l]
            d = image[y_h * img_width + x_h]

            resized = a * (1 - x_weight) * (1 - y_weight) + \
                    b * x_weight * (1 - y_weight) + \
                    c * y_weight * (1 - x_weight) + \
                    d * x_weight * y_weight

            new_image[k,...,i] = resized.reshape(height, width)
            
    if frames == 1:
        new_image = new_image[0]
        

    return new_image



def ccm_3x3(image):
    ccm = np.array([[ 1.32601113,  0.16077133,  0.10276821],
           [ 0.07652649,  1.65069942,  0.05202971],
           [ 0.7758613 , -0.56706419,  2.42086586]])
    
    out_im = (image[...,0:3] - image[...,3][...,np.newaxis]).dot(ccm)

    return out_im

def ccm_3x4(image):
    ccm = np.array([[ 0.76031811,  0.19460622, -0.09200754, -0.04863701],
       [-0.30808756,  1.67370372, -0.08039811, -0.73159016],
       [ 0.2734654 , -0.53701519,  2.24788416, -1.26116684]])
    
    out_im = (image).dot(ccm.transpose())

    return out_im

def ccm2_3x4(image):
    loaded = scipy.io.loadmat('../visualization/color_correction.mat')
    
    ccm = loaded['ccm34']
    ccm = ccm/np.max(ccm)
    out_im = (image).dot(ccm.transpose())

    return out_im

def ccm2_3x3(image):
    loaded = scipy.io.loadmat('../visualization/color_correction.mat')
    
    ccm = loaded['ccm33']
    ccm = ccm/np.max(ccm)
    
    out_im = (image[...,0:3] - image[...,3:]).dot(ccm.transpose())

    return out_im


import cv2
def white_balance(img):
    img = img.astype('float32')
    img = img[...,0:3]
    
    if len(img.shape)==4:
        nframes = img.shape[0]
        
    else:
        nframes = 1
        img = img[np.newaxis]
        
        
    img_out = np.empty_like(img)
        
    for i in range(0, nframes):
        result = cv2.cvtColor(img[i], cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        img_out[i] = result
        
    if nframes == 1:
        img_out = img_out[0]
        
    return img_out

def mult(image, a=2.5):
    return image*a

def clip(image):
    return np.clip(image, 0, 1)

def gamma(image, gamma = 2.2):
    return image**(1/gamma)

def process(image, transforms= [bayer_bilinear, ccm_3x4, clip, gamma]):
    for i in transforms:
        image = i(image)
    return image

def process_contrast(img, param= 0.2):
    luminance = np.clip(rgb2lum(img),0.0,1.1)
    luminance = luminance[...,np.newaxis]
    contrast_lum = np.cos(np.pi*luminance)*0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    

    return lerp(img, contrast_image, param)

def lerp(a, b, l):
    return (1 - l) * a + l * b

def rgb2lum(image):
    return 0.27 * image[..., 0] + 0.67 * image[..., 1] + 0.06 * image[..., 2]

def saturation(img, param=np.array([.5,.5,.5])):
    img = img.astype('float32')
    img = np.minimum(img, np.ones_like(img))

    if len(img.shape)==3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif len(img.shape)==4:
        hsv_new = np.empty_like(img)
        for i in range(0,img.shape[0]):
            hsv_new[i] = cv2.cvtColor(img[i].astype('float32'), cv2.COLOR_RGB2HSV)
        hsv = hsv_new
    else:
        print('invalid image shape')
    
    enhanced_s = hsv[...,1] + (1 - hsv[...,1]) * (0.5 - np.abs(0.5 - hsv[...,2])) * 0.8
    
    
    new_HSV = np.stack([hsv[...,0], enhanced_s, hsv[...,2]], -1)
    
    if len(img.shape)==3:
        full_color = cv2.cvtColor(new_HSV, cv2.COLOR_HSV2RGB)
    elif len(img.shape)==4:
        full_color_new = np.empty_like(img)
        for i in range(0,img.shape[0]):
            full_color_new[i] = cv2.cvtColor(new_HSV[i], cv2.COLOR_HSV2RGB)
        full_color = full_color_new
    img_param = 1-param
    return img*img_param +full_color*param 

def contrast(img, param):
    luminance = np.clip(rgb2lum(img),0.0,1.1)[...,np.newaxis]
    contrast_lum = np.cos(np.pi*luminance)*0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    

    return lerp(img, contrast_image, param)