"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import os
import glob
from random import shuffle, randint

class voc_seg_dataset_reader:
    path = ""
    class_mappings = ""
    files = []
    images = []
    annotations = []
    test_images = []
    test_annotations = []
    batch_offset = 0
    epochs_completed = 0
    # voc images mean
    VOC_IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


    def __init__(self, vocdevkit_path, max_pages=None, crop=False, crop_size=[1000,1000], test_size=20, divisor_32= True):
        """
        Initialize a file reader for the DeepScores classification data
        :param records_list: path to the dataset
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        """
        print("Initializing pascal VOC segmentation Batch Dataset Reader...")
        self.path = vocdevkit_path
        self.max_pages = max_pages
        self.crop = crop
        self.crop_size = crop_size
        self.test_size = test_size
        self.divisor_32 = divisor_32

        images_list = []
        with open(self.path + "/ImageSets/Segmentation/trainval.txt") as f:
            images_list = f.read().splitlines()


        #shuffle image list
        shuffle(images_list)

        if max_pages is None:
            max_pages = len(images_list)


        if max_pages > len(images_list):
            print("Not enough data, only " + str(len(images_list)) + " available")
            print(" At " + self.path)
            import sys
            sys.exit(1)

        if test_size >= max_pages:
            print("Test set too big ("+str(test_size)+"), max_pages is: "+str(max_pages))
            print(" At " + self.path)
            import sys
            sys.exit(1)

        print("Splitting dataset, train: "+str(max_pages-test_size)+" images, test: "+str(test_size)+ " images")
        test_image_list = images_list[0:test_size]
        train_image_list = images_list[test_size:max_pages]

        # test_annotation_list = [image_file.replace("/images_png/", "/pix_annotations_png/") for image_file in test_image_list]
        # train_annotation_list = [image_file.replace("/images_png/", "/pix_annotations_png/") for image_file in train_image_list]

        self._read_images(test_image_list,train_image_list)

    def _read_images(self,test_image_list,train_image_list):

        dat_train = [self._transform(filename) for filename in train_image_list]
        for dat in dat_train:
            self.images.append(dat[0])
            self.annotations.append(dat[1])
        self.images = np.array(self.images)

        self.annotations = np.array(self.annotations)

        print("Training set done")
        dat_test = [self._transform(filename) for filename in test_image_list]
        for dat in dat_test:
            self.test_images.append(dat[0])
            self.test_annotations.append(dat[1])
        self.test_images = np.array(self.test_images)

        self.test_annotations = np.array(self.test_annotations)

        print("Test set done")

    # def translate_voc(self, a):
    #     return [np.where(np.sum(self.colours_list == a, axis=1) == 3)[0][0]]

    def _transform(self, filename):
        image = misc.imread(self.path+"/JPEGImages/"+filename+".jpg")
        image = image - self.VOC_IMG_MEAN
        annotation = misc.imread(self.path+"/SegmentationClass/pre_encoded/"+filename+".png")
        annotation = np.expand_dims(annotation,-1)
        if not image.shape[0:2] == annotation.shape[0:2]:
            print("input and annotation have different sizes!")
            import sys
            import pdb
            pdb.set_trace()
            sys.exit(1)

        if self.crop:
            coord_0 = randint(0, (image.shape[0] - self.crop_size[0]))
            coord_1 = randint(0, (image.shape[1] - self.crop_size[1]))

            image = image[coord_0:(coord_0+self.crop_size[0]),coord_1:(coord_1+self.crop_size[1])]
            annotation = annotation[coord_0:(coord_0 + self.crop_size[0]), coord_1:(coord_1 + self.crop_size[1])]

        if self.divisor_32:
            # pad image such that 32 is a divisor of height and with
            h = int(np.ceil(image.shape[0] / 32.0))
            w = int(np.ceil(image.shape[1] / 32.0))
            img_canv = np.zeros((h*32,w*32,image.shape[2]), dtype=image.dtype)
            ann_canv = np.zeros((h * 32, w * 32, annotation.shape[2]), dtype=annotation.dtype)
            img_canv[0:image.shape[0], 0:image.shape[1]] = image
            ann_canv[0:image.shape[0], 0:image.shape[1]] = annotation
            return[img_canv, ann_canv]

        return [image, annotation]

    # from PIL import Image
    # im = Image.fromarray(image)
    # im.show()
    # im = Image.fromarray(annotation)
    # im.show()


    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def get_test_records(self):
        return self.test_images, self.test_annotations

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]



if __name__ == "__main__":
    data_reader = voc_seg_dataset_reader("/Users/tugg/datasets/DeepScores")
