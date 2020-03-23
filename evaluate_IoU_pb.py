from __future__ import print_function
from utils import read_labeled_image_list
import os
import math
import time
import cv2
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import json

# ------------------------------------------------------------------------------------------------------------------ #
# Входные параметры
# ------------------------------------------------------------------------------------------------------------------ #

#
data = "bdd"
#
data_path = ""
#
eval_list_path = "/media/data/ObjectDetectionExperiments/Datasets/12_Cityscapes/lists/BDD_val.txt"
#
dir = './models/UNet'
# Префикс имени моделей, которые мы будем использовать
model_prefix = "unet_"
# Список суффиксов имен моделей, которые мы будем использовать
model_suffix = ["best", "99ep"]
# Коэффициеты преорзования CamVid
coef_for_enet = [[10,-1], [1, -1], [1.67,-1], [0, -1], [0.2,-1], [1.34,-1],
                 [1.17, 1], [0.5,-1], [1.45, -1], [1.1,-1], [1.64, 1.1], [21.25,-1]]
# Формируем название выходного файла
output_file = "txt_result/" + model_prefix + "_" + data + ".txt"

# ------------------------------------------------------------------------------------------------------------------ #
# Выполнение инференса
# ------------------------------------------------------------------------------------------------------------------ #

def only_resize(img, label):

    img = cv2.resize(img, (inp_w, inp_h))
    label = cv2.resize(label, (inp_w, inp_h), interpolation=cv2.INTER_NEAREST)
    return img, label

def resize_with_padding(img, label, asp):

    img_w = img.shape[1]
    img_h = img.shape[0]
    aspi = img_w/img_h
    if asp*0.8 < aspi < asp*1.25:
        img = cv2.resize(img, (inp_w, inp_h))
        label = cv2.resize(label, (inp_w, inp_h), interpolation=cv2.INTER_NEAREST)
    else:
        if asp > aspi:
            new_w = asp * img_h
            img = cv2.copyMakeBorder(img, 0, 0, 0, new_w - img_w, cv2.BORDER_CONSTANT, value=0) # Паддинг по ширине
            label = cv2.copyMakeBorder(label, 0, 0, 0, new_w - img_w, cv2.BORDER_CONSTANT, value=255)

        else:
            new_h = int(img_w / asp)
            img = cv2.copyMakeBorder(img, 0, new_h - img_h, 0, 0, cv2.BORDER_CONSTANT, value=0) # Паддинг по высоте
            label = cv2.copyMakeBorder(label, 0, new_h - img_h, 0, 0, cv2.BORDER_CONSTANT, value=255)

        img = cv2.resize(img, (inp_w, inp_h))
        label = cv2.resize(label, (inp_w, inp_h), interpolation=cv2.INTER_NEAREST)
    return img, label

def only_padding(img, label, inp_w, inp_h):

    img_w = img.shape[1]
    img_h = img.shape[0]
    img = cv2.copyMakeBorder(img, 0, inp_h - img_h, 0, inp_w - img_w, cv2.BORDER_CONSTANT, value=0)
    label = cv2.copyMakeBorder(label, 0, inp_h - img_h, 0, inp_w - img_w, cv2.BORDER_CONSTANT, value=255)
    return img, label

def evaluate_one_image(label, outimg, num_classes, true_pos, false_pos, false_neg, model, coef_for_enet):


    for i in range(num_classes):

        if 'enet' in model:
            label_mask = np.equal(label, int((i + 1) * coef_for_enet[i][0]))  # Приведение соответствия CamVid
            label_mask += np.equal(label, int((i + 1) * coef_for_enet[i][1]))
        else:
            label_mask = np.equal(label, i)

        img_mask = np.equal(outimg, i)
        true_pos_mask = img_mask * label_mask
        false_neg_mask = img_mask * (~true_pos_mask)

        true_pos[i] = true_pos[i] + np.sum(true_pos_mask)
        false_pos[i] = false_pos[i] + (np.sum(label_mask) - np.sum(true_pos_mask))
        false_neg[i] = false_neg[i] + np.sum(false_neg_mask)

    return true_pos, false_pos, false_neg


result_list = []

for j in model_suffix:

    name = model_prefix + str(j)
    model_path = os.path.join(dir, name+".pb")

    print("Preparing model... ", end='')
    with open(os.path.join(dir, name+".json")) as f:
        json_dict = json.load(f)

    input = json_dict["input_node"]
    output = json_dict["output_node"]
    num_classes = json_dict["nb_classes"]
    inp_w = json_dict["input_w"]
    inp_h = json_dict["input_h"]
    asp = inp_w/inp_h

    with tf.Session() as sess:
        print("load graph")
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        input = sess.graph.get_tensor_by_name(input)
        output = sess.graph.get_tensor_by_name(output)


        image_list, label_list = read_labeled_image_list(data_path, eval_list_path)

        true_pos = np.zeros(num_classes)
        false_pos = np.zeros(num_classes)
        false_neg = np.zeros(num_classes)
        IoU = [0] * num_classes
        t_sum = 0

        for i, img_path in enumerate(image_list):
            img = cv2.imread(img_path)
            label = cv2.imread(label_list[i])
            img, label = resize_with_padding(img, label, asp) # ///ресайз и паддинг\\\
            one_ch_label = label[:, :, 0]
            t = timer()
            # with tf.Session() as sess:
            y_out = sess.run(output, feed_dict={input: img})
            t_sum = t_sum + (timer() - t)
            if i == 0:
                t_sum = 0
            # Далее постобработка уже. Пригодится и для других сетей
            # [1, 1024, 2048, 19] -> [1024, 2048, 19]
            outimg_lables = np.squeeze(y_out)
            outimg = np.argmax(outimg_lables, axis=2)

            true_pos, false_pos, false_neg = evaluate_one_image(one_ch_label, outimg, outimg_lables.shape[2], true_pos, false_pos, false_neg, name, coef_for_enet)
            print(j, ":", i, "/", len(image_list))

    tf.reset_default_graph()
    # Средние значения по всем изображниям
    true_pos = true_pos / len(image_list)
    false_pos = false_pos / len(image_list)
    false_neg = false_pos / len(image_list)
    mean_time = t_sum / len(image_list)
    FPS = 1/mean_time

    for i, one_class_true in enumerate(true_pos):
        IoU[i] = one_class_true/(one_class_true + false_pos[i] + false_neg[i])

    if 'enet' in name:
        IoU.pop()
        label_names = json_dict["label_names"]
        label_names.pop()
        for i, label_name in enumerate(label_names):
            print(label_name, ' = ', "%.5f" % IoU[i])
    else:
        print('0.road = ', "%.5f" %          IoU[0])
        print('1.sidewalk = ', "%.5f" %      IoU[1])
        print('2.building = ', "%.5f" %      IoU[2])
        print('3.wall = ', "%.5f" %          IoU[3])
        print('4.fence = ', "%.5f" %         IoU[4])
        print('5.pole = ', "%.5f" %          IoU[5])
        print('6.traffic light = ', "%.5f" % IoU[6])
        print('7.traffic sign = ', "%.5f" %  IoU[7])
        print('8.vegetation = ', "%.5f" %    IoU[8])
        print('9.terrain = ', "%.5f" %       IoU[9])
        print('10.sky = ', "%.5f" %          IoU[10])
        print('11.person = ', "%.5f" %       IoU[11])
        print('12.rider = ', "%.5f" %        IoU[12])
        print('13.car = ', "%.5f" %          IoU[13])
        print('14.truck = ', "%.5f" %        IoU[14])
        print('15.bus = ', "%.5f" %          IoU[15])
        print('16.train = ', "%.5f" %        IoU[16])
        print('17.motorcycle = ', "%.5f" %   IoU[17])
        print('18.bicycle = ', "%.5f" %      IoU[18])

    mIoU = np.mean(IoU)

    print('mIoU = ', "%.5f" % mIoU)
    print('FPS = ', FPS)

    line = "{}: mIoU = {}; FPS = {}\n".format(name, mIoU, FPS)
    result_list.append(line)

with open(output_file, "w") as f:
    f.writelines(result_list)