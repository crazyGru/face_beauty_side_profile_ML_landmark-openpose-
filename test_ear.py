import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter
from imutils import paths
import cv2


input_size = 368

landmark_size = 27
ear_part_num = [20, 15, 15, 5]
ear_threshold = 0.5

r_size = 368



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


model = tf.keras.models.load_model('saved_model/saved_model_openpose_ear_v1.h5', compile=False)

pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])


image_folder = 'input'
imagePaths = sorted(list(paths.list_images(image_folder)))

for i, imagePath in enumerate(imagePaths):
    frame = cv2.imread(imagePath)
    h,w = frame.shape[:2]

    # if h > w:
    #     frame_origin = frame[:w,:,:]
    #     output_w = w
    # else:
    #     frame_origin = frame[:,:h,:]
    #     output_w = h
        
    # frame = frame_origin
    frame = cv2.resize(frame, (368, 368))
    # frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ear-detect---------------------------------------------------------------
    result = pred([np.expand_dims(image, axis=0)/255.])[0]
    result[result < ear_threshold] = 0 # threshold setting
    result = tf.image.resize(result, [r_size, r_size])
    result = apply_blur(result, landmark_size).numpy()
    result = np.argmax(result.reshape(-1,landmark_size), axis=0)


    prev = []
    for i, idx in enumerate(result):
        x, y = idx%r_size/r_size*w, idx//r_size/r_size*w
        prev.append([int(x),int(y)])

    print(len(prev), prev, result)

    for target in prev:
        img = cv2.circle(frame, target, 2, (0,0,255), -1)

    cv2.imshow("VFrame", img)
    cv2.waitKey(0)  