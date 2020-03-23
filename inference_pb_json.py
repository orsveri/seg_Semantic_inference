"""
Скрипт с примером кода для выполнения вывода (вывода, предсказывания) на изображении с использованием нейронной сети,
сохранённой в формате pb-файла
"""
from __future__ import print_function
import os
from timeit import default_timer as timer
import cv2
import numpy as np
import tensorflow as tf
import json

# Функция для декодирования выхода сети в цветную маску
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

# ------------------------------------------------------------------------------------------------------------------ #
# Входные параметры
# ------------------------------------------------------------------------------------------------------------------ #

# Список с изображениями для вывода
images = ['./test_images/input/test1.png', './test_images/input/test2.png', './test_images/input/test3.png',
          './test_images/input/bdd1.jpg', './test_images/input/bdd2.jpg', './test_images/input/bdd3.jpg']

# Папка с сохранёнными конфигурациями модели
dir = './models/UNet'
# Название конфигурации <name>. Будут загружены <name>.pb и <name>.json из <dir>.
name = "unet_99ep"
# Папка, куда сохранять получившиеся картинки (сохранение закомментировано)
save_dir = './output/'

# ------------------------------------------------------------------------------------------------------------------ #
# Выполнение инференса
# ------------------------------------------------------------------------------------------------------------------ #

model_path = os.path.join(dir, name+".pb")

print("Preparing model... ", end='')
t = timer()

# Загружаем json с мета-информацией
with open(os.path.join(dir, name+".json")) as f:
    json_dict = json.load(f)

input = json_dict["input_node"]
output = json_dict["output_node"]
num_classes = json_dict["nb_classes"]
inp_w = json_dict["input_w"]
inp_h = json_dict["input_h"]

# Для запуска модели необходимо задать сессию тензорфлоу
with tf.Session() as sess:
    print("load graph")
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    # Загружаем модель
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    input = sess.graph.get_tensor_by_name(input)
    output = sess.graph.get_tensor_by_name(output)

    t_delta = timer() - t
    print("{} s".format(t_delta), end="\n\n")

    for i in images:
        t = timer()

        img_path = i
        img = cv2.imread(img_path)
        img_shape = img.shape[:2]
        # TODO: smart resize
        inp_img = cv2.resize(img, (inp_w, inp_h))

        t1 = timer()

        y_out = sess.run(output, feed_dict={input: inp_img})

        # Далее постобработка уже. Пригодится и для других сетей
        #  [1, H, W, num_classes] -> [H, W, num_classes]
        outimg = np.squeeze(y_out)

        t_delta1 = timer() - t1

        # [H, W, num_classes] -> [H, W, 3]
        outclasses = decode_labels_pb(mask=outimg,
                                      num_classes=num_classes,
                                      label_colours=json_dict["label_colours"])

        t_delta = timer() - t
        print("Time for predict image: {} s,".format(t_delta1))
        print("Time for predict image with pre- and postprocessing: {} s.".format(t_delta), end="\n\n")

        cv2.imshow("orig", img)
        result = cv2.resize(outclasses, (img_shape[1], img_shape[0]))
        cv2.imshow("mask", result)
        # Сохранение
        # cv2.imwrite(os.path.join(save_dir, os.path.basename(i)), result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()