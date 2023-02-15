import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


class DatasetManipulation:
    def __init__():
        pass

    def Grayscale(dataset, shape=(8,8,1), normalize=True) -> np.array:
        images = []
        for img in tqdm(dataset, desc="Grayscaling images"):
            tmp_image = np.array([x[:,0] for x in img])
            tmp_image = np.reshape(tmp_image, shape)
            images.append(tmp_image)
        return (np.array(images)/255 if normalize else np.array(images))

    def RemoveDuplicates(dataset_x: np.array, dataset_y: np.array):
        uniques, uniq_indices = np.unique(dataset_x, axis=0, return_index=True)
        indices = dataset_y[uniq_indices]
        return uniques, indices, uniq_indices

    def NormalizeByCategory(dataset_x, dataset_y):
        cats = len(set(dataset_y)) + 1
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

    def Histogram(dataset_y):
        tmp_bins = [0] + list(set(dataset_y))
        counts, bins = np.histogram(dataset_y, bins=np.array(tmp_bins))
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()