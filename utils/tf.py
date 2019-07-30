# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: tensorflow's utils.
"""
from enum import IntEnum
import numpy as np
import tensorflow as tf


class ImgEnh(IntEnum):
    _BASE = 0b1
    LR_FILP     = (_BASE << 0)
    UD_FLIP     = (_BASE << 1)
    ROT         = (_BASE << 2)
    ROT_90      = (_BASE << 3)
    CROP        = (_BASE << 4)
    ALL         = 0b11111


def _tf_random_rotate90(image, rotate_prob=0.9):
    rotated = tf.image.rot90(image)
    rand = tf.random_uniform([], minval=0, maxval=1)
    return tf.cond(tf.greater(rand, rotate_prob), lambda: image, lambda: rotated)


def img_enh(batch_img_ts, ops=ImgEnh.ALL):
    if isinstance(batch_img_ts, np.ndarray):
        batch_img_ts = tf.convert_to_tensor(batch_img_ts)
    if ops & ImgEnh.ROT_90 != 0:
        batch_img_ts = _tf_random_rotate90(batch_img_ts)
    def flip(img):
        if ops & ImgEnh.LR_FILP != 0:
            img = tf.image.random_flip_left_right(img)
        if ops & ImgEnh.UD_FLIP != 0:
            img = tf.image.random_flip_up_down(img)
        return img
    batch_img_ts = tf.map_fn(flip, batch_img_ts)
    if ops & ImgEnh.CROP != 0:
        ori_shape = [int(batch_img_ts.shape[i]) for i in range(len(batch_img_ts.shape))]
        # print(ori_shape)
        batch_img_ts = tf.random_crop(batch_img_ts, [batch_img_ts.shape[0], round(int(batch_img_ts.shape[1]) * 0.5), 
                                                   round(int(batch_img_ts.shape[2]) * 0.5), batch_img_ts.shape[-1]])
        batch_img_ts = tf.image.resize_images(images=batch_img_ts, size=ori_shape[1:3])
    if ops & ImgEnh.ROT:
        random_angles = tf.random.uniform(shape=(tf.shape(batch_img_ts)[0],), minval=-np.pi/10, maxval=np.pi/10)
        batch_img_ts = tf.contrib.image.transform(
                batch_img_ts,
                tf.contrib.image.angles_to_projective_transforms(
                    random_angles, tf.cast(batch_img_ts.shape[1], tf.float32), tf.cast(batch_img_ts.shape[2], tf.float32)
                ))
    return batch_img_ts


if __name__=='__main__':
    a = np.zeros([3,4,4,2])
    a = np.array(range(a.size)).reshape(a.shape)
    print(a[0,:,:,0])
    print(a[0,:,:,1])
    a = tf.convert_to_tensor(a)
    print(a.shape[-1])
    # exit()
    sess = tf.Session()
    # with tf.Session() as sess:
    e = img_enh(a, ops=ImgEnh.CROP)
    e = e.eval()
    print(e[0,:,:,0])
    print(e[0,:,:,1])

