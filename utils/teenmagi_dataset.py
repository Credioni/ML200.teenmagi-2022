import math
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
import pickle
import os

class TeenmagiDataset(Sequence):

    def __init__(self,  categories=[i+1 for i in range(1000)],
                        batch_size=16,
                        n_valid=50,
                        zero_percentage=0.0,
                        sparse=False,
                        denoiser=False,
                        use_area=False,

                        master_model=False,
                 ):

        super(TeenmagiDataset, self).__init__()
        self.train_iset = None
        self.valid_iset = None
        self.nimage_set = None

        self.category_reindexing    = None

        self.training_x = None
        self.training_y = None

        self.length = int(10e3)
        self.zero_percentage = zero_percentage
        self.batch_size = batch_size
        self.sparse=sparse
        self.denoiser = denoiser
        self.use_area = use_area
        self.master_model = master_model
        self.output_size = None

        self.n_valid = n_valid
        self.categories = categories

        # These are areas which are mixed. Seen in correlation map 
        self.combine_areas = [
                 #(1,   215),
                 #(229, 249),
                 #(250, 294),
                 (359, 369),
                 (389, 439),
                 #(439, 499),
                 #(679, 729)
                ]

        
        #path = os.getcwd()
        #print(path)
        #path = path + "\\training_manager.dat"
        path = "data/training_manager.dat"

        with open(path, 'rb') as pickleFile:
            data = pickle.load(pickleFile)
            [self.training_x, self.training_y, _, self.nimage_set] = data
            self.training_y = self.training_y.astype(int)
            #[self.training_x, self.training_y, self.validation_x, self.nimage_set]

        self.train_iset, self.valid_iset, self.category_reindexing =\
            self.__generate(categories)

        if len(self.categories) != 1000:
            n_zero = len(self.categories) * self.zero_percentage / (1 - self.zero_percentage)
            n_zero = math.ceil(n_zero) + 1
            self.categories = [0 for i in range(n_zero)] + self.categories

        #print("teenmagi_dataset categories:", self.categories)
        self.ncats = [0 for i in range(len(self.categories) + 1)]
        self.output_size = len(set(self.category_reindexing))

    def __generate(self, categories):
        category_reindexing = np.array([i for i in range(1001)])

        if self.use_area:
            for area in self.combine_areas:
                (start, end) = area
                iarea = np.array([i for i in range(start, end)])
                category_reindexing[iarea] = category_reindexing[start]
                last = 1
                for i, x in enumerate(category_reindexing):
                    if abs(x - last) > 1:
                        category_reindexing[i] = last + 1
                        last = last + 1
                    else:
                        last = x

        UPPER = 2000
        for i, ci in enumerate(categories):
            tmp = np.where(category_reindexing==ci)[0]
            category_reindexing[tmp] = UPPER + i + 1
        tmp = np.where(category_reindexing<=UPPER)[0]
        category_reindexing[tmp] = 0
        tmp = np.where(category_reindexing>UPPER)[0]
        category_reindexing[tmp] -= UPPER

        category_size = len(set(category_reindexing))
        train_iset = [[] for i in range(category_size)]
        valid_iset = [[] for i in range(category_size)]

        # Creating index datasets
        for ci, rci in enumerate(category_reindexing):
            cimg_i = np.where(self.training_y==ci)[0]
            train_iset[rci].append(cimg_i[self.n_valid:])
            valid_iset[rci].append(cimg_i[:self.n_valid])

        for i in range(len(train_iset)):
            train_iset[i] = np.concatenate(train_iset[i], axis=0).flatten()
            valid_iset[i] = np.concatenate(valid_iset[i], axis=0).flatten()

        return train_iset, valid_iset, category_reindexing

    def validation_set(self, only_categories=False):
        valid_x, valid_y = [], []

        for i, c_iset in enumerate(self.valid_iset):
            if (only_categories and not (i+1 in categories)) or len(c_iset) == 0:
                continue

            c_imgs = self.training_x[c_iset]
            valid_x.append(c_imgs)

            ci = self.category_reindexing[i]
            valid_y.append( np.full(len(c_imgs), ci) )

        valid_x = np.concatenate(valid_x, axis=0)
        valid_y = np.concatenate()

        return ( valid_x, valid_y )

    def get_output_size(self):
        return self.output_size

    def info(self):
        print("Categories:", self.categories)
        print("Length:", self.length)

        print("n categories:", self.ncats)

    def reset_stats(self):
        self.ncats = [0 for i in range(len(self.categories) + 1)]

    def get(self, use_valid_iset=False, ci=None):
        iset = self.valid_iset if use_valid_iset else self.train_iset

        re_ci = list(set(self.category_reindexing))

        if ci == None:
            if len(iset[0]) == 0:
                re_ci = re_ci[1:]
            ci = np.random.choice(re_ci)

        img_i = np.random.choice(iset[ci])

        y = np.zeros( (len(re_ci), 1) )
        y[ci] = 1
        x = self.training_x[img_i]

        return (x, y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        re_ci = list(set(self.category_reindexing))

        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            (x, y) = self.get()
            batch_x.append(x)
            batch_y.append(y)

        if self.denoiser:
            denoiser_batch_y = []
            for i, c in enumerate(batch_y):
                ci = c
                if not self.sparse:
                    ci = np.argmax(ci)
                    if len(self.categories) == 1000:
                        ci += 1

                if np.random.choice([True, False]) or True:
                    # choose x image
                    img = batch_x[i]
                else:
                    # choose category normalized image
                    img = self.nimage_set[0][ci]
                    #img = np.reshape(self.nimage_set[0][ci], (8,8,1))
                denoiser_batch_y.append(img)
            batch_y = denoiser_batch_y

        return np.array(batch_x), np.array(batch_y)


if __name__ == "__main__":
    path = os.getcwd()
    print("working path:", path)
    teenmagi_dataset = TeenmagiDataset(
        #categories=[i+1 for i in range(10)],
        batch_size=1,
        n_valid=100,
        # zero_percentage=0.1,
        sparse=False,
        master_model=True,
    )

    for i in range(1):
        teenmagi_dataset[0]

    (valid_x, valid_y) = teenmagi_dataset.validation_set()
    print("Ended")
