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
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        
        return pred

    # Возвращает маску с n_classes каналов
    def predict_one_raw(self, input_img):
        img_shape = input_img.shape[:2]
        inp_img = cv2.resize(input_img, (self.inp_w, self.inp_h))
        y_out = self.sess.run(self.output, feed_dict={self.input: inp_img})
        #  [1, H, W, num_classes] -> [H, W, num_classes]
        outimg = np.squeeze(y_out)
        result = cv2.resize(outimg, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        return result

    # Возвращает трехканальную цветную маску
    def predict_one(self, input_img):
        assert self.label_colours, "Цвета классов не заданы в json-файле!"
        img_shape = input_img.shape[:2]
        inp_img = cv2.resize(input_img, (self.inp_w, self.inp_h))
        y_out = self.sess.run(self.output, feed_dict={self.input: inp_img})
        #  [1, H, W, num_classes] -> [H, W, num_classes]
        outimg = np.squeeze(y_out)
        # [H, W, num_classes] -> [H, W, 3]
        outclasses = self.decode_labels_pb(mask=outimg)
        result = cv2.resize(outclasses, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        return result

    def predict_video(self, video_path, output_path, show=True):
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open webcam. Make sure you video_path is an integer!"))
    
        # Compute aspect ratio of video
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
        # Define the codec and create VideoWriter object
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(vidw), int(vidh)))
    
        while True:
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                return
        
            res_image = self.predict_one(orig_image)
        
            if show:
                cv2.imshow("YOLOv2 result", res_image)
        
            if output_path:
                out.write(res_image)
        
            pressedKey = cv2.waitKey(10)
            if pressedKey == 27:  # ESC key
                break
    
        out.release()

# ------------------------------------------------------------------------------------------------------------------ #
# Входные параметры
# ------------------------------------------------------------------------------------------------------------------ #

# Список с изображениями для вывода
images = ['./test_images/input/test1.png', './test_images/input/test2.png', './test_images/input/test3.png',
          './test_images/input/bdd1.jpg', './test_images/input/bdd2.jpg', './test_images/input/bdd3.jpg',
          './test_images/input/cityscapes2.png']

# Папка с сохранёнными конфигурациями модели
dir = './models/UNet'
# Название конфигурации <name>. Будут загружены <name>.pb и <name>.json из <dir>.
name = "unet_99ep_TEST"
# Папка, куда сохранять получившиеся картинки (сохранение закомментировано)
save_dir = './test_images/output/{}'.format(name)

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
    img = cv2.imread(img_path)
    
    res_img = prd.predict_one(input_img=img)

    t1 = timer()
    t_delta1 = t1 - t

    print("Время предсказания: {} s,".format(t_delta1))

    cv2.imshow("orig", img)
    cv2.imshow("mask", res_img)
    
    # Сохранение
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(i)), res_img)

    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    
del prd
