import os
import numpy as np
import cv2
from PIL import Image


class ImageResizer:
    def __init__(self, input_dir, output_dir, file_ext='png'):
        self.__input_dir = input_dir
        self.__output_dir = output_dir
        self.__filelist = os.listdir(self.__input_dir)
        self.__imgs = []
        self.__set_imgs(file_ext=file_ext)

    def get_input_list(self):
        return self.__filelist

    def get_images(self):
        return self.__imgs

    def __set_imgs(self, file_ext='png'):
        for item in self.__filelist:
            if item.find('.'+file_ext) != -1:
                self.__imgs.append(item)

    def transform(self, save=True):
        total_image = len(self.__imgs)
        index = 0

        for name in self.__imgs:
            img = Image.open('%s/%s' % (self.__input_dir, name))
            img_array = np.array(img)
            img_resize = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(img_resize)
            if save:
                img.save('%s%s' % (self.__output_dir, name))

            print(name + '   ' + str(index) + '/' + str(total_image))
            index = index + 1
