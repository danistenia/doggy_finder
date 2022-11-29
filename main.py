'''Finding Similar Dogs'''

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import tqdm
import os

class RandomImg:
    def __init__(self, folder):
        self.folder = folder
        self.selected_dog = None
        self.selected_dogs = []

    def random_folder(self):
        list_dir = os.listdir(self.folder)
        rand_idx = random.choice(range(len(list_dir)))
        return rand_idx, list_dir[rand_idx]

    def random_img(self):
        _, name_folder = self.random_folder()
        #print(name_folder)
        folder = os.listdir(os.path.join(self.folder, name_folder))
        rand_idx_img = random.choice(range(len(folder)))
        #img = plt.imread(os.path.join(self.folder, name_folder, folder[rand_idx_img]))
        #plt.imshow(img)
        self.selected_dog = os.path.join(self.folder, name_folder, folder[rand_idx_img])

    def random_images(self):

        list_dir = os.listdir(self.folder)
        for path in list_dir:
            path_image_list = os.listdir(os.path.join(self.folder, path))
            rand_idx = random.choice(range(len(path_image_list)))
            self.selected_dogs.append(os.path.join(self.folder, path, path_image_list[rand_idx]))

    def execute_img_imgs(self):
        self.random_img()
        self.random_images()


class DoggyFinder:
    def __init__(self, input_img, list_images):
        self.img_size = (160, 160)
        self.img_shape = self.img_size + (3, )
        self.input_img = input_img
        self.list_images = list_images
        self.model = None
        self.single_img = None
        self.vectors = []
        self.idxs = []
        self.matrix_d = None
        self.imgs_similar = None

    def create_model(self):
        
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.img_shape,
                                                    include_top=False,
                                                    weights='imagenet')

        base_model.trainable = False
        
        feature_vector_layer = tf.keras.layers.GlobalAveragePooling2D(name="feature_extractor")
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        inputs = tf.keras.Input(shape=self.img_shape)
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = feature_vector_layer(x)
        outputs = x
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    
    def __prepare_img(self, img=None):
        if img == None:

            treated_img = tf.keras.preprocessing.image.load_img(self.input_img, 
                                                        target_size = self.img_size)
            treated_img = np.array([tf.keras.preprocessing.image.img_to_array(treated_img)])
        
        else:
            treated_img = tf.keras.preprocessing.image.load_img(img, 
                                                        target_size = self.img_size)
            treated_img = np.array([tf.keras.preprocessing.image.img_to_array(treated_img)])

        return treated_img

    def single_predict_fv(self):
        treated_img = self.__prepare_img(self.input_img)
        feature_vector_img = self.model.predict(treated_img)
        self.single_img = feature_vector_img
        #return feature_vector_img
    
    def batch_predict_fv(self):

        for image in tqdm.tqdm(self.list_images):
            
            treated_img = self.__prepare_img(img=image)
            pred = self.model(treated_img)
            self.vectors.append(pred)

    def __eucledian_distance(self, x, y):       
        eucl_dist = np.linalg.norm(x - y)
        return eucl_dist

    def d_matrix(self):
        D = [self.__eucledian_distance(self.single_img, y) for y in self.vectors]
        self.matrix_d = D

    def top_n(self, n):
        self.d_matrix()
        similar_images_idx = np.argpartition(self.matrix_d, n)
        list_idx = list(similar_images_idx)[:n]
        self.imgs_similar = [self.list_images[i] for i in list_idx]
    
    def plot_similar(self, n):
        self.top_n(n)
        plt.imshow(plt.imread(self.input_img))
        plt.title(self.input_img)
        plt.axis('off')
        rows = 3
        cols = int(n / rows)
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15,4))
        
        for i, ax in enumerate(axs.flatten()):

            img = plt.imread(self.imgs_similar[i])
            plt.sca(ax)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Parecido {i+1}')
        
        plt.show()
            
        

    
random_class = RandomImg(folder=r'images')
random_class.execute_img_imgs()

img, list_imgs = random_class.selected_dog, random_class.selected_dogs

doggy = DoggyFinder(input_img= img, list_images=list_imgs)
doggy.create_model()
doggy.single_predict_fv()
doggy.batch_predict_fv()

#n = 10
#doggy.top_n(n=n)
doggy.plot_similar(n=10)


#print(doggy.imgs_similar)
#print(doggy.matrix_d)



