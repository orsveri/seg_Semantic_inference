import numpy as np
import cv2
import tensorflow as tf
import os

def decode_labels_pb(mask, num_classes, label_colours):

    # TODO: smart argmax, skip pixels with low confidence
    outclasses = np.argmax(mask, axis=2)

    outshape = list(outclasses.shape)
    outshape.append(3)

    color_table = label_colours
    color_mat = np.array(color_table, dtype=np.uint8)

    onehot_output = (np.arange(num_classes) == outclasses[...,None]).astype(int)

    onehot_output = np.reshape(onehot_output, (-1, num_classes))
    pred = np.matmul(onehot_output, color_mat)
    pred = np.rint(pred).astype(np.uint8)
    pred = np.reshape(pred, outshape)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    return pred


def read_labeled_image_list(data_dir, data_list):
    f = open(data_list, 'r')

    images = []
    masks = []
    for line in f:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")

        image = os.path.join(data_dir, image)
        mask = os.path.join(data_dir, mask)
        mask = mask.strip()

        if not tf.gfile.Exists(image):
            raise ValueError('Failed to find file: ' + image)

        if not tf.gfile.Exists(mask):
            raise ValueError('Failed to find file: ' + mask)

        images.append(image)
        masks.append(mask)

    return images, masks