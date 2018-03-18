from scipy.signal import butter, lfilter
import numpy as np
import pickle
import random
import os
from sklearn.model_selection import KFold
from sklearn.utils import shuffle



class DataSet(object):
    def __init__(self,
                 loadDatabase_savepoint=False,
                 loadName="",
                 saveDatabase=False,
                 saveName="",
                 newDataBaseName="",
                 data_window = 1000,
                 testSize=50,
                 nbKFold=10):

        if(loadDatabase_savepoint):
            with open(loadName) as tmp:
                print("Loading ready-to-use dataset: " + loadName)
                dataset, dataset_gtBoxes, dataset_gtBoxes_labels, testset, testset_gtBoxes, \
                testset_gtBoxes_labels, dataset_train_fold, dataset_valid_fold, \
                nbKFold, mean, max, min = pickle.load(tmp)

        else:
            #1 ---- Load the raw database
            print("Loading raw database:" + newDataBaseName )
            dataset, dataset_gtBoxes, dataset_gtBoxes_labels = loadRawDataset(newDataBaseName)


            #1.1 -- transform labels to softmax format format
            print("Transforming labels to fit softmax ouputs")
            dataset_gtBoxes_labels = transformLabels(dataset_gtBoxes_labels)

            #2 ---- Filtering and normalizing the data
            print("Normalizing input signals")
            dataset, mean, max, min = normalize_data(dataset)
            self._mean = mean
            self._max = max
            self._min = min

            #3 ---- Shuffle the data
            print("Shuffling data before doing test split")
            dataset, dataset_gtBoxes, dataset_gtBoxes_labels = shuffleDatabase(dataset, dataset_gtBoxes, dataset_gtBoxes_labels)

            #4 ---- Split for test
            print("Test split of size " + str(testSize))
            testset = dataset[:testSize]
            testset_gtBoxes = dataset_gtBoxes[:testSize]
            testset_gtBoxes_labels = dataset_gtBoxes_labels[:testSize]
            dataset = dataset[testSize:]
            dataset_gtBoxes = dataset_gtBoxes[testSize:]
            dataset_gtBoxes_labels = dataset_gtBoxes_labels[testSize:]


            #5 ---- Create datasets
            print("Creating dataset with window of length: " + str(data_window))
            dataset, dataset_gtBoxes_labels, dataset_gtBoxes \
                = createWindowDataBase(dataset, dataset_gtBoxes, dataset_gtBoxes_labels ,data_window,
                                           fg_overlap=0.8, bg_overlap=0.3)

            #6 ---- Filter data
            #dataset = np.reshape(dataset, (-1, 9, 1000))
            #dataset = butterfilter_data(dataset)

            #6 ---- Generate folds
            print("Generating " + str(nbKFold) + " folds")
            x = dataset
            kf = KFold(n_splits=nbKFold)
            kf.get_n_splits(x)
            dataset_train_fold = []
            dataset_valid_fold = []
            for train_index, test_index in kf.split(x):
                dataset_train_fold.append(train_index)
                dataset_valid_fold.append(test_index)

            #8 ---- Save everything
            if(saveDatabase):
                print("Saving ready-to-use dataset to " + saveName)
                with open(saveName, 'w') as tmp:
                    pickle.dump([dataset,
                                 dataset_gtBoxes,
                                 dataset_gtBoxes_labels,
                                 testset,
                                 testset_gtBoxes,
                                 testset_gtBoxes_labels,
                                 dataset_train_fold,
                                 dataset_valid_fold,
                                 nbKFold,
                                 mean,
                                 max,
                                 min], tmp)
            print("Done")


        self._train_data_KFold = []
        self._train_label_KFold = []
        self._train_boxes_KFold = []
        self._valid_data_KFold = []
        self._valid_label_KFold = []
        self._valid_boxes_KFold = []
        self._index_in_epoch = 0
        self._dataset = np.array(dataset)
        self._dataset_gtBoxes = np.array(dataset_gtBoxes)
        self._dataset_gtBoxes_labels = np.array(dataset_gtBoxes_labels)
        self._testset = testset
        self._testset_gtBoxes = testset_gtBoxes
        self._testset_gtBoxes_labels = testset_gtBoxes_labels
        self._dataset_train_fold = dataset_train_fold
        self._dataset_valid_fold = dataset_valid_fold
        self._KFold = KFold
        self._mean = mean
        self._max = max
        self._min = min

    def init_fold_data(self, fold_num):

        indexes = self._dataset_train_fold[fold_num]
        self._train_data_KFold = self._dataset[indexes]
        self._train_boxes_KFold = self._dataset_gtBoxes[indexes]
        self._train_label_KFold = self._dataset_gtBoxes_labels[indexes]
        mix = np.arange((self._train_data_KFold).__len__())
        np.random.shuffle(mix)
        self._train_data_KFold = self._train_data_KFold[mix]
        self._train_label_KFold = self._train_label_KFold[mix]
        self._train_boxes_KFold = self._train_boxes_KFold[mix]

        indexes = self._dataset_valid_fold[fold_num]
        self._valid_data_KFold = self._dataset[indexes]
        self._valid_label_KFold = self._dataset_gtBoxes_labels[indexes]
        self._valid_boxes_KFold = self._dataset_gtBoxes[indexes]

    def init_fold_data_per_label(self, fold_num, wanted_label):
        self.init_fold_data(fold_num)

        #get only elements equal to label
        index_train = [i for i,item in enumerate(self._train_label_KFold) if item[wanted_label]==1]
        index_valid = [i for i,item in enumerate(self._valid_label_KFold) if item[wanted_label]==1]

        self._train_data_KFold = self._train_data_KFold[index_train]
        self._train_label_KFold = self._train_label_KFold[index_train]
        self._train_boxes_KFold = self._train_boxes_KFold[index_train]
        self._valid_data_KFold = self._valid_data_KFold[index_valid]
        self._valid_label_KFold = self._valid_label_KFold[index_valid]
        self._valid_boxes_KFold = self._valid_boxes_KFold[index_valid]

    def save_norm_variable(self, saveName):
        with open(saveName, 'w') as tmp:
            pickle.dump([self._mean, self._max, self._min], tmp)


def loadRawDataset(filename):
    if os.path.isfile(filename):
        with open(filename) as tmp:
            dataset, dataset_gtBoxes, dataset_gtBoxes_labels = pickle.load(tmp)
    else:
        print "File not found"
        dataset, dataset_gtBoxes, dataset_gtBoxes_labels = [0,0,0]
    return dataset, dataset_gtBoxes, dataset_gtBoxes_labels

def shuffleDatabase(dataset, dataset_gtBoxes, dataset_gtBoxes_labels):
    dataset, dataset_gtBoxes, dataset_gtBoxes_labels = shuffle(dataset, dataset_gtBoxes, dataset_gtBoxes_labels)
    return dataset, dataset_gtBoxes, dataset_gtBoxes_labels


def createWindowDataBase(dataset, dataset_gtBoxes, dataset_gtBoxes_labels ,
                             data_window, stride = 100, fg_overlap = 0.7, bg_overlap = 0.3):

    newDatabaseScale = []
    new_all_labels = []
    new_bound_boxes = []
    scale = data_window

    for y in np.arange(len(dataset)):
        data = dataset[y]
        dataLen = len(data[0])

        gtboxes = dataset_gtBoxes[y]
        labels = dataset_gtBoxes_labels[y]

        input_list = []
        bound_box_list = []
        labels_list = []
        for z in np.arange((dataLen - scale) / stride):
            start = z * stride
            end = start + scale

            tmp = np.reshape(data[:, start:end], (scale, 9))
            label, bound_box = IoU_per_section(start, end, gtboxes, labels, fg_overlap, bg_overlap)

            if(label != [-1,-1,-1]):
                input_list.append(tmp)
                bound_box_list.append(bound_box)
                labels_list.append(label)


        #Keep only necessary data
        indexes = keep_ratio_per_label(labels_list, bg_ratio = 1.5 )

        input_list = np.reshape(input_list, (-1, scale, 9))
        input_list = input_list[indexes]
        input_list = input_list.astype(np.float32)

        bound_box_list = np.reshape(bound_box_list, (-1, 3))
        bound_box_list = bound_box_list[indexes]
        bound_box_list = bound_box_list.astype(np.int16)

        labels_list = np.reshape(labels_list, (-1,3))
        labels_list = labels_list[indexes]
        labels_list = labels_list.astype(np.int8)

        for i in np.arange(len(indexes)):
            newDatabaseScale.append(input_list[i])
            new_all_labels.append(labels_list[i])
            new_bound_boxes.append(bound_box_list[i])

    return newDatabaseScale, new_all_labels, new_bound_boxes


def keep_ratio_per_label(label_list, bg_ratio, min_bg = 5):
    fgIndexes = []
    bgIndexes = []
    for i in np.arange(len(label_list)):
        a = label_list[i]
        if(label_list[i][0]==1):    #background
            bgIndexes.append(i)
        else:                       #foreground
            fgIndexes.append(i)

    nb_fg = len(fgIndexes)
    nb_bg = nb_fg*bg_ratio
    nb_bg = max(nb_bg,min_bg)


    if(nb_bg<len(bgIndexes)):
        bgIndexes = random.sample(bgIndexes,int(nb_bg))


    return fgIndexes + bgIndexes


def IoU_per_section(start, end, gtboxes, labels, fg_overlap = 0.7, bg_overlap = 0.3):
    label = [1,0,0] # we start by assuming it's background
    bound_box = [0,0,start]

    for i in np.arange(len(labels)):
        gt_start = gtboxes[i][0]
        gt_end = gtboxes[i][1]
        #Area of overlap calcul
        interArea = min(gt_end, end) - max(gt_start,start)

        iou = float(interArea)/float(gt_end-gt_start)

        if(iou>fg_overlap):
            label = labels[i]
            bound_box = [gt_start-start,gt_end-start, start]

        elif(iou<fg_overlap and iou>bg_overlap): #we don't want to train on ambigous sections
            label = [-1,-1,-1]


    return label, bound_box



def transformLabels(all_labels): #only 3 classes
    labels = []
    for i in np.arange(len(all_labels)):
        itemlabels = []
        for y in np.arange(len(all_labels[i])):
            if all_labels[i][y] == 1:
                itemlabels.append([0,1,0]) #insertion
            elif all_labels[i][y] == 2:
                itemlabels.append([0,0,1]) #placing
            elif all_labels[i][y] == 3:
                itemlabels.append([1,0,0]) #background
            elif all_labels[i][y] == 4:
                itemlabels.append([1,0,0]) #background
            else:
                itemlabels.append([1,0,0]) #background
        labels.append(itemlabels)
    return np.array(labels)


def normalize_data(data):
    nb_data = len(data)
    mean= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    max= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    min = np.array([99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0])
    for index in range(nb_data):
        for dim in range(len(data[index])):
            mean[dim] += data[index][dim].mean()
            if data[index][dim].max() > max[dim]:
                max[dim] = data[index][dim].max()
            if data[index][dim].min() < min[dim]:
                min[dim] = data[index][dim].min()


    mean = mean/(nb_data)
    for index in range(nb_data):
        for dim in range(len(data[index])):
            data[index][dim] = ((data[index][dim] - min[dim]) / (max[dim] - min[dim]))

    return data, mean, max, min


def normalize_data_from_previous(data,min,max):
    nb_data = len(data)

    for index in range(nb_data):
        for dim in range(len(data[index])):
            data[index][dim] = ((data[index][dim] - min[dim]) / (max[dim] - min[dim]))

    return data

def butterfilter_data(data):
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 125.0
    lowcut = 0.0
    highcut = 30

    #filteredData = np.ndarray(data.shape, dtype=float)
    filteredDataBase = []

    for i in np.arange(data.__len__()):
        filteredData = []
        for y in np.arange(data[i].__len__()):
            filteredData.append(butter_bandpass_filter(data[i][y], lowcut, highcut, fs, order=4))
        filteredDataBase.append(filteredData)


    return filteredDataBase


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



if __name__ == '__main__':
    DataSet(newDataBaseName="Insertion_database_16March2018.dataset", saveDatabase=True, saveName="test.dat")
    #DataSet(loadDatabase_savepoint=True, loadName="23fev.dataset")
