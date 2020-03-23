"""
Скрипт с примером кода для выполнения вывода (вывода, предсказывания) на изображении с использованием нейронной сети,
сохранённой в формате pb-файла
"""
from __future__ import print_function
import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import json
from PIL import Image


class Predictor():
    
    def __init__(self, model_pb, meta_json, quiet=True):
        # Загружаем json с мета-информацией
        with open(meta_json) as f:
            json_dict = json.load(f)
        
        input = json_dict["input_node"]
        output = json_dict["output_node"]
        self.num_classes = json_dict["nb_classes"]
        self.inp_w = json_dict["input_w"]
        self.inp_h = json_dict["input_h"]
        self.label_colours = json_dict["label_colours"] if "label_colours" in json_dict else None

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        
        # Для запуска модели необходимо задать сессию тензорфлоу
        self.sess = tf.Session(config=config)
        if not quiet: print("Загрузка модели...", end='')
        with tf.gfile.GFile(model_pb, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
        # Загружаем модель
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
        self.input = self.sess.graph.get_tensor_by_name(input)
        self.output = self.sess.graph.get_tensor_by_name(output)
        
        if not quiet: print("Готово!")
    
    # Функция для декодирования выхода сети в цветную маску
    def decode_labels_pb(self, mask):
        # TODO: smart argmax, skip pixels with low confidence
        outclasses = np.argmax(mask, axis=2)
        
        outshape = list(outclasses.shape)
        outshape.append(3)
        
        color_table = self.label_colours
        color_mat = np.array(color_table, dtype=np.uint8)
        
        onehot_output = (np.arange(self.num_classes) == outclasses[..., None]).astype(int)
        
        onehot_output = np.reshape(onehot_output, (-1, self.num_classes))
        pred = np.matmul(onehot_output, color_mat)
        pred = np.rint(pred).astype(np.uint8)
        pred = np.reshape(pred, outshape)
        
        return pred

    # Возвращает трехканальную цветную маску
    def predict_one(self, input_img):
        assert self.label_colours, "Цвета классов не заданы в json-файле!"
        img_size = input_img.size

        input_img = input_img.resize((self.inp_w, self.inp_h), Image.ANTIALIAS)
        input_img = np.asarray(input_img)
        input_img = np.flip(input_img, axis=-1)
        
        y_out = self.sess.run(self.output, feed_dict={self.input: input_img})
        #  [1, H, W, num_classes] -> [H, W, num_classes]
        outimg = np.squeeze(y_out)
        # [H, W, num_classes] -> [H, W, 3]
        outclasses = self.decode_labels_pb(mask=outimg)
        
        result = Image.fromarray(outclasses)
        result = result.resize(img_size, Image.NEAREST)
        return result


# ------------------------------------------------------------------------------------------------------------------ #
# Входные параметры
# ------------------------------------------------------------------------------------------------------------------ #

# Список с изображениями для вывода
images = ['./test_images/input/test1.png', './test_images/input/test2.png', './test_images/input/test3.png',
          './test_images/input/bdd1.jpg', './test_images/input/bdd2.jpg', './test_images/input/bdd3.jpg']

# Папка с сохранёнными конфигурациями модели
dir = './models/PSPNet'
# Название конфигурации <name>. Будут загружены <name>.pb и <name>.json из <dir>.
name = "psp1"
# Папка, куда сохранять получившиеся картинки (сохранение закомментировано)
save_dir = './output/'

# ------------------------------------------------------------------------------------------------------------------ #
# Выполнение инференса
# ------------------------------------------------------------------------------------------------------------------ #

model_path = os.path.join(dir, name + ".pb")
json_path = os.path.join(dir, name + ".json")

t = timer()

prd = Predictor(model_pb=model_path, meta_json=json_path)

t_delta = timer() - t
print("Загрузка, инициализация модели - {} с".format(t_delta), end="\n\n")

for i in images:
    t = timer()

    img_path = i
    
    img = Image.open(img_path)
    
    res_img = prd.predict_one(input_img=img)

    t1 = timer()
    t_delta1 = t1 - t

    print("Время предсказания: {} s,".format(t_delta1))

    img.show()
    res_img.show()
    
del prd
