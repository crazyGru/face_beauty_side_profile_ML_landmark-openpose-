import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter

import cv2

#---setting------------------------------#

# recording setting
recording = False
recording_time = 25

# model input size
input_size = 368#200

# ear model
landmark_size = 27
ear_part_num = [20, 15, 15, 5]
ear_threshold = 0.5


# resize setting for coordinate correction
r_size = 368#200

# capture size (Default = 620x480)
IM_W = 1280
IM_H = 720

# output size (Default = 620x480)
o_size_w = 720
o_size_h = 720

#----------------------------------------#

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def apply_blur(img, landmark):
    blur = _gaussian_kernel(5, 2.5, landmark, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
    return img[0]

color_list = [(0,255,0), (255,51,0), (255,204,0), (255,204,0)]

model = tf.keras.models.load_model('saved_mode/saved_model_openpose_ear_v1.h5', compile=False)

pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])


cut_w_r = (IM_W - IM_H) // 2
cut_w_l = IM_W - cut_w_r



image = cv2.imread('./ex.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
    
    # ear-detect---------------------------------------------------------------
result = pred([np.expand_dims(image, axis=0)/255.])[0]
result[result < ear_threshold] = 0 # threshold setting
result = tf.image.resize(result, [r_size, r_size])
result = apply_blur(result, landmark_size).numpy()
result = np.argmax(result.reshape(-1,landmark_size), axis=0)

print(result)

prev_xy = [[],[],[],[]]
for i, idx in enumerate(result):
        x, y = idx%r_size/r_size*o_size_w, idx//r_size/r_size*o_size_h
        # if x < 1 or y < 1 : continue
        # if i > 49: prev_xy[3].append([int(x),int(y)]); continue
        # if i > 34: prev_xy[2].append([int(x),int(y)]); continue
        # if i > 19: prev_xy[1].append([int(x),int(y)]); continue
        prev_xy[0].append([int(x),int(y)])
    
for i, xy in enumerate(prev_xy):
        if len(xy)==ear_part_num[i]:cv2.polylines(image, [np.asarray(xy)], False , color_list[i], 2)
    # -------------------------------------------------------------------------
    
    # guide line
cv2.ellipse(image, (o_size_w//2,o_size_h//2), (o_size_w//5,o_size_h//3), 0, 0, 360, (0, 255, 0), 1)
cv2.imshow("VFrame", image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()