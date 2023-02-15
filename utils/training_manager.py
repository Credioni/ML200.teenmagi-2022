import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pandas as pd
import glob

import time

import random
from random import shuffle
import math

from os import walk
import pickle

from tempfile import TemporaryFile

from IPython.display import clear_output
from tqdm.notebook import tqdm
#from tqdm import tqdm


class TrainingSetManager():
    def __init__(self, datapath="data/"):
        # (grayscale_img, cat_index) for denoiser 
        self.image_set = None
        self.nimage_set = None

        # Grayscaled images
        self.training_x = None
        self.training_y = None
        self.validation_x = None

        self.test_set = None

        self.path = datapath

        self.category_sizes = []
        for i in range(0,1001):
            asd = np.where(self.training_y==i)[0]
            self.category_sizes.append(len(asd))

        for i in range(2):
            try:
                with open(self.path + "training_manager.dat", 'rb') as pickleFile:
                    data = pickle.load(pickleFile)

                [self.training_x, self.training_y, self.validation_x, self.nimage_set] = data
                self.nimage_set = list(zip(self.nimage_set[0], self.nimage_set[1]))
                print("Loaded grayscaled datasets")
                break
            except:
                if i==1:
                    print("Couldnt load grayscaled sets.")
                    return
                
                print("Trying to update training manager data")
                [self.training_x, self.training_y, self.validation_x, self.nimage_set] = self.update()


    def update(self):
        try:
            with open(self.path + "training_x.dat", 'rb') as pickleFile:
                training_x = pickle.load(pickleFile)
            with open(self.path + "training_y.dat", 'rb') as pickleFile:
                training_y = pickle.load(pickleFile)
            with open(self.path + "validation_x.dat", 'rb') as pickleFile:
                validation_x = pickle.load(pickleFile)
            
            # deleting broken training set img
            del training_y[216805]
            del training_x[216805]
            print("Training data loaded")
        except:
            print("\nUpdate: couldnt load datasets")
            print("Manually load and add them to data/* folder")
            print("https://www.kaggle.com/competitions/teenmagi-2022/data \n")
            return False

        #self.training_x   = self.grayscale_images(self.training_x)
        #self.validation_x = self.grayscale_images(self.validation_x)
        training_x   = self.grayscale_images(training_x)
        validation_x = self.grayscale_images(validation_x)

        tmp = []
        for y in training_y:
            tmp.append(int(y))
        training_y = np.array(tmp)

        norm_imgs, std = self.get_normalized_images(self.training_x, self.training_y)

        tmp_data = [training_x, training_y, validation_x, [norm_imgs, std]]

        print("Saving data as training_manager.dat")
        with open('data/training_manager.dat', 'wb') as handle:
            pickle.dump(tmp_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return [training_x, training_y, validation_x, [norm_imgs, std]]

    def gen_training_set(self,
                             #category_indices=[i for i in range(1001)],
                             categories = [i+1 for i in range(1000)],

                             n_valid=50,
                             n_test=50,

                             category_sizes=1300,
                             category_zero_size=None, # Take all imgs

                             from_logits=True,
                             denoiser=False,
                             n_augmentation: int=0,
                             plot=False,
                        )-> (np.array, np.array, np.array):
        """
        category_indices : Constaints list of new indices starting from 1 to n
         n_valid=150 : positive int; taken from end of list
         n_test=50   : positive int; taken from beginning of list
         category_sizes=1300 : cropped size of each categories
         category_padding=True  : if categories are padding to correspond to category_sizes
         category_noise=False   : -//- applying noises images
         padding_noise_percentage=0.5, # Padding procent over noise
         denoiser=False,
        """
        ########
        category_padding=True
        ########

        #category_zero_size = category_zero_size if category_zero_size != None else category_sizes

        category_reindexing = np.array([0 for i in range(1000)])
        for i, ci in enumerate(categories):
            category_reindexing[ci-1] = i + 1

        tmp_x = np.array(self.training_x, copy=True)
        tmp_y = np.array(self.training_y, copy=True)

        train_izero = np.array([])
        train_iset = [np.array([]) for i in range(1000)]

        valid_iset, test_iset = np.array([]),  np.array([])

        # Creating index datasets
        print("\nGenerating set:")
        for ci in tqdm(range(1000), ascii=True):
            # Getting category image indices; [0] indices, [1] data type
            cimg_i = np.where(tmp_y==(ci+1))[0]
            # Creating index sets
            test_iset = np.concatenate((test_iset, cimg_i[:n_test]), axis=0)
            valid_iset = np.concatenate((valid_iset, cimg_i[n_test:n_test + n_valid]), axis=0)
            # Shuffle training so zero category is randomized orderly each time
            ci_train_iset = cimg_i[n_test + n_valid:]
            p = np.random.permutation(len(ci_train_iset))
            ci_train_iset = ci_train_iset[p]
            train_iset[ci] = np.concatenate((train_iset[ci], ci_train_iset), axis=0)

        # Padding
        # -> outputting same size of arrays for train_iset
        #print("\nPadding:")
        for ci in range(1000):
            #ci = i + 1
            cimg_i = train_iset[ci]
            delta =  category_sizes - len(cimg_i)

            if delta > 0:
                # Getting random images indices for padding
                if category_padding:
                    n_padding = delta
                    # How many times to tile cimg_i to cut inaf of data
                    n_tile = math.ceil( delta / len(cimg_i) )
                    # Create padding and shuffle before applying
                    cimg_delta = np.tile(cimg_i, n_tile)
                    p = np.random.permutation(len(cimg_delta))
                    cimg_pad = cimg_delta[p][:n_padding]

                    train_iset[ci] = np.concatenate((train_iset[ci], cimg_pad), axis=0)
            else:
                train_iset[ci] = train_iset[ci][:category_sizes]

        ## used in augmentation and recombination
        tmp_train_iset = np.array(train_iset, dtype=int, copy=True)

        # Data augmentation
        train_set_x_aug = []
        train_set_y_aug = []
        if n_augmentation > 0:
            print("\nAugmentation:")
            for i, cimgs_iset in enumerate(tqdm(tmp_train_iset, ascii=True)):
                ci = i + 1
                p = np.random.permutation(len(cimgs_iset))
                cimgs_iset = cimgs_iset[p][:n_augmentation]
                cimgs = tmp_x[cimgs_iset]
                r = np.random.rand()
                if r < 1/3:
                    tmp_imgs = self.crop_pad(cimgs, 1)
                elif r < 2/3:
                    tmp_imgs = self.push(cimgs, 1)
                else:
                    tmp_imgs = self.crop_pad(cimgs, 1)
                    tmp_imgs = self.push(tmp_imgs, 1)
                train_set_x_aug.append(tmp_imgs)
                train_set_y_aug.append( np.full(len(tmp_imgs), ci) )
            train_set_x_aug = np.concatenate(train_set_x_aug, axis=0)
            train_set_y_aug = np.concatenate(train_set_y_aug, axis=0)

        # Recombine categories
        print("\nCollecting samples for sets:")
        test = list(set(category_reindexing))
        for i, ci in enumerate(tqdm(set(category_reindexing), ascii=True)):
            re_ic = np.where(category_reindexing==ci)[0]

            # Getting desired category's indices
            cat_indices = tmp_train_iset[re_ic]
            # Rolling indices so each category's indices gets ...
            cat_indices = cat_indices.T
            cat_icut = np.concatenate(cat_indices)
            # Cutting and combining
            if ci == 0:
                train_izero = cat_icut
            elif ci > 0:
                train_iset[ci -1] = cat_icut[:category_sizes]

        # Reindexing
        for i, ci in enumerate(category_reindexing):
            ci_tmp_y = np.where(tmp_y==(i + 1))[0]
            tmp_y[ci_tmp_y] = ci

        # Set y;labels accordingly
        if not from_logits:
            tmp_y_nlogits = np.zeros((len(tmp_y), len(categories)))
            for i, ci in enumerate(categories):
                re_ic = np.where(tmp_y==ci)[0]
                ci_pos = np.full(len(re_ic), i)
                tmp_y_nlogits[re_ic, ci_pos] = 1
            tmp_y = tmp_y_nlogits
        elif denoiser:
            tmp_y_denoiser = np.zeros( (len(tmp_y), 8, 8, 1) )
            print("\nDenoiser labels:")
            for i, ci in enumerate(tqdm(categories, ascii=True)):
                re_ic = np.where(tmp_y==ci)[0]
                tmp_y_denoiser[re_ic] = np.reshape(self.nimage_set[ci][0], (8,8,1))

            tmp_y = tmp_y_denoiser

        # Removing old categories from categories
        train_iset = train_iset[:len(categories)]

        # List to numpy array
        train_iset = np.array(train_iset, dtype=int)

        # Stacking
        train_iset = np.hstack(train_iset).astype(int)
        valid_iset = np.hstack(valid_iset).astype(int)
        test_iset  = np.hstack(test_iset).astype(int)


        if category_zero_size != None and len(train_izero) > 0:
            train_iset = np.concatenate((train_iset, train_izero[:category_zero_size]), axis=0)
        elif len(train_izero) > 0:
            train_iset = np.concatenate((train_izero, train_iset), axis=0)


        train_set = [tmp_x[train_iset], tmp_y[train_iset]]
        valid_set = [tmp_x[valid_iset], tmp_y[valid_iset]]
        test_set  = [tmp_x[test_iset], tmp_y[test_iset]]

        # Add data augmenatation
        if n_augmentation:
            train_set[0] = np.concatenate((train_set[0], train_set_x_aug), axis=0)
            train_set[1] = np.concatenate((train_set[1], train_set_y_aug), axis=0)

        # Shuffling training set
        p = np.random.permutation(len(train_set[0]))
        train_set[0], train_set[1] = train_set[0][p], train_set[1][p]

        if plot and not denoiser:
            self.plot( [train_set[1], valid_set[1], test_set[1]],
                        ["train_set", "valid_set", "test_set"],
                        np.arange(1002)
            )
        return (train_set, valid_set, test_set)

    def plot(self, data_ys, names=None, bins=1000):

        n = len(data_ys)
        print(n)
        c_distribution = []

        for i in range(n):
            x = np.histogram(data_ys[i], bins=bins)[0]
            c_distribution.append(x)

        plt.figure(figsize=(10, 5))
        for i in range(n):
            # display reconstruction
            bx = plt.subplot(1, 3, i + 1)
            plt.yscale('symlog')
            if names != None:
                plt.title(names[i])
            dist = c_distribution[i]
            cut_dist = np.where(dist !=0)[0][-1]
            plt.plot(dist)
            #bx.get_xaxis().set_visible(False)
            #bx.get_yaxis().set_visible(False)

        plt.show()

    def gen_denoiser_set(self):
        pass

    def gen_noise_images(self, category:int, n=100, shape=(8,8,1)):
        images = []
        for i in range(n):
            (tmp_img, tmp_std) = self.nimage_set[category]
            noise_img = np.random.normal(tmp_img, tmp_std)
            noise_img = np.clip(noise_img, 0, 1)
            #noise_img = np.array(noise_img)
            images.append(np.reshape(noise_img, shape))
        return np.array(images)

    def grayscale_images(self, img_set, shape=(8,8,1), normalize=True):
        images = []
        for img in tqdm(img_set, desc="Grayscaling images"):
            tmp_image = np.array([x[:,0] for x in img])
            tmp_image = np.reshape(tmp_image, shape)
            if normalize:
                tmp_image=tmp_image/255
            images.append(tmp_image)
        return np.array(images)

    def get_normalized_images(self, set_img, set_img_cat):
        cats = len(set(set_img_cat)) + 1
        norm_images = [np.matrix(np.zeros((8,8))) for i in range(cats)]
        elements = [0 for i in range(cats)]

        # Calculate mean
        for idx, img in tqdm(enumerate(set_img), desc="Calculating means"):
            img_class = set_img_cat[idx]
            tmp_image = img
            tmp_image = np.reshape(tmp_image, (8,8))
            norm_images[img_class] += np.matrix(tmp_image)
            elements[img_class] += 1

        for i in range(cats):
            norm_images[i] = norm_images[i] / elements[i]

        # Calculate variance
        variances = [np.zeros((8,8)) for i in range(cats)]
        for iimg, img in tqdm(enumerate(set_img), desc="Calculating variances"):
            img_cat = set_img_cat[iimg]
            mean_img = np.array(norm_images[img_cat])

            for irow, row in enumerate(img):
                for iobj, obj in enumerate(row):
                    tmp_var = np.power((obj - mean_img[irow][iobj]), 2 ) / len(norm_images)
                    variances[img_cat][irow][iobj] = tmp_var

        return np.array(norm_images), np.sqrt(np.array(variances))

    def validate_model(self, model):
        results = model.predict(self.validation_x)
        results_arg = [np.argmax(x) for x in results]
        results_df = pd.DataFrame({"Id" : [ x + 1 for x in range(len(results_arg))], "Class" : results_arg})
        results_df = results_df.set_index(["Id"])
        return results_df, results_arg

    def crop_pad(self, imgs, percentage=0.1, shape=(8,8,1)):
        tmp_imgs = imgs.copy()
        for i, img in enumerate(tmp_imgs):
            tmp_img = np.reshape(img, (8,8))
            if percentage <= np.random.rand(1) or True:
                if np.random.rand(1) <= 0.5:
                    tmp_img = tmp_img[1:7,0:8]
                    tmp_img = np.pad(tmp_img, ((1,1), (0,0)), mode='constant', constant_values=(0, 0))
                else:
                    tmp_img = tmp_img[0:8,1:7]
                    tmp_img = np.pad(tmp_img, ((0,0), (1,1)), mode='constant', constant_values=(0, 0))
            tmp_imgs[i] = np.reshape(tmp_img, shape)
        return tmp_imgs

    def push(self, imgs, percentage=0.1, shape=(8,8,1)):
        dirs = [((1,0), (0,0)), ((0,1), (0,0)),
               ((0,0), (1,0)), ((0,0), (0,1)),
              ]
        tmp_imgs = imgs.copy()
        for i, img in enumerate(tmp_imgs):
            tmp = np.reshape(img, (8,8))
            if percentage <= np.random.rand(1) or True:
                direction = random.randint(0, 3)
                tmp = np.pad(tmp, dirs[direction], mode='constant', constant_values=(0, 0))
                if direction==0:
                    tmp = tmp[0:8,0:8]
                elif direction==1:
                    tmp = tmp[1:9,0:8]
                elif direction==2:
                    tmp = tmp[0:8,0:8]
                else:
                    tmp = tmp[0:8,1:9]
            tmp_imgs[i] = np.reshape(tmp, shape)
        return tmp_imgs

    def plot_history(history):
        acc = history.history['accuracy']
        loss = history.history['loss']

        epochs_range = range(len(history.history['loss']))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='upper left')
        plt.ylim(0,1)
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()




if __name__ == "__main__":
    training_set_manager = TrainingSetManager(datapath="../data/")

    categories=[i for i in range(100, 150)]
    print("categories len:", len(categories))

    (train_set, valid_set, test_set) =\
        training_set_manager.gen_training_set(
                                                #categories=categories,
                                                plot=True,
                                                category_sizes=2000,
                                            )


#
