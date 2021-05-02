import pandas as pd
from skimage.feature import greycomatrix, greycoprops
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import random
import cv2 as cv
from statistics import mean
from matplotlib import pyplot as plt


def convert_images_to_textural_characteristics(root, file_name='input_params.csv'):
    """ Function reads images and convert them to Haralick's Dataframe using GLCM matrix.

    :param root:        Path to dir with disease folders.
    :param file_name:   Name of the file to save data in.
    :return:            Frame of textural characteristics of images.
    """
    masks = [[132, 0, 0], # red
             [204, 208, 193], # grey
             [245, 228, 103], # light yellow
             [54, 46, 34], # brown
             [182, 146, 82], # light brown
             [0, 255, 0]] # light green
    r_coef, g_coef, b_coef = (0.3, 0.59, 0.11)

    offset = [20]
    angle = [-np.pi / 2]
    levels = 8

    df_data = []  # saving parameters Haralick's

    for i, folder_dir in enumerate(os.listdir(root)):
        if folder_dir[0] == ".":
            continue

        # print("Folder: ", folder_dir)
        for _, filename in enumerate(os.listdir(root + folder_dir)):
            path = root + folder_dir + "/" + filename
            # print(path)
            if filename[0] == ".":
                continue

            image = cv.imread(path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = np.array(image, dtype=np.int64)
            masked_images = np.zeros((6,) + image.shape[:-1])
            for i, mask in enumerate(masks):
                masked_images[i] = (r_coef * (1 - (np.abs(image[:, :, 0] - mask[0])) / max(mask[0], 255 - mask[0]))
                                    + g_coef * (1 - (np.abs(image[:, :, 1] - mask[1])) / max(mask[1], 255 - mask[1]))
                                    + b_coef * (1 - (np.abs(image[:, :, 2] - mask[2])) / max(mask[2], 255 - mask[2])))
            masked_images = np.array(masked_images * (levels - 1), dtype=np.uint16)
            glcms = np.zeros((len(masks), levels, levels, 1, 1))

            for i in range(len(masks)):
                glcms[i] = greycomatrix(masked_images[i], offset, angle, levels=levels, symmetric=False, normed=True)

            temp_dict = {}
            for k in range(len(glcms)):
                correlation = greycoprops(glcms[k], 'correlation')[0, 0]
                contrast = greycoprops(glcms[k], 'contrast')[0, 0]
                homogeneity = greycoprops(glcms[k], 'homogeneity')[0, 0]
                energy = greycoprops(glcms[k], 'energy')[0, 0]
                dissimilarity = greycoprops(glcms[k], 'dissimilarity')[0, 0]
			
                temp_dict.update({'correlation{}'.format(k + 1): correlation,
                                  'contrast{}'.format(k + 1): contrast,
                                  'homogeneity{}'.format(k + 1): homogeneity,
                                  'energy{}'.format(k + 1): energy,
                                  'dissimilarity{}'.format(k + 1): dissimilarity})

            temp_dict.update({'class': int(folder_dir)})			
            df_data.append(temp_dict)

    pd.DataFrame(df_data).to_csv(file_name, sep='\t', index=False)
    # return df_data


def generate_haralick_params(df_data, SKO, iteration_count, averaged_elements):
    """Function creates new dataframe of Haralick's params.
    
    :param df_data:                  Frame of Haralick's params.
    :param SKO:                      The noise value.
    :param iteration_count:          Count of iterations of generating parameter sets from the averaged group.
    :param averaged_elements:        Count of averaging params.
    :return:                         New dataframe of Haralick's params.
    """
    def get_hapalick_params(df_data_new, df_data, i, averaged_elements, SKO):
        """Helper function for generating Haralick's params.

        :param df_data_new:             Datarame for gererated Haralick's params.
        :param df_data:                 Initial frame of Haralick's params.
        :param i:                       Number of the current image.
        :param averaged_elements:       Count of averaging params.
        :param SKO:                     The noise value.
        :return:
        """
        cl = df_data[i]['class']
        temp_dict = {}
        for k in range(6):
            correlation = averaging_over_n_images(df_data, i, averaged_elements, 'correlation{}'.format(k + 1), SKO)
            contrast = averaging_over_n_images(df_data, i, averaged_elements, 'contrast{}'.format(k + 1), SKO)
            homogeneity = averaging_over_n_images(df_data, i, averaged_elements, 'homogeneity{}'.format(k + 1), SKO)
            energy = averaging_over_n_images(df_data, i, averaged_elements, 'energy{}'.format(k + 1), SKO)
            dissimilarity = averaging_over_n_images(df_data, i, averaged_elements, 'dissimilarity{}'.format(k + 1), SKO)

            temp_dict.update({'correlation{}'.format(k + 1): correlation,
                              'contrast{}'.format(k + 1): contrast,
                              'homogeneity{}'.format(k + 1): homogeneity,
                              'energy{}'.format(k + 1): energy,
                              'dissimilarity{}'.format(k + 1): dissimilarity})
        temp_dict.update({'class': cl})

        df_data_new.append(temp_dict)
        return df_data_new

    def averaging_over_n_images(df_data, i, averaged_elements, key, SKO):
        """Helper function for generating Haralick's params.
        Function averages n images to find the mathematical expectation of textural params and add some noize to it.

        :param df_data:                 Frame of Haralick's params.
        :param i:                       Number of the current image.
        :param averaged_elements:       Count of averaging params.
        :param key:                     Name of Haralick's param.
        :param SKO:                     The noise value.
        :return:                        New parameter value.
        """
        if SKO != 0:
            low = 1 - SKO
            up = 1 + SKO
            SKO = random.choice(np.arange(low, up, 0.0001))
        else:
            SKO = 1
        if averaged_elements == 1:
            return SKO * df_data[i][key]
        if averaged_elements == 2:
            return SKO * (df_data[i - 1][key] + df_data[i][key]) / averaged_elements
        elif averaged_elements == 3:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key]) / averaged_elements
        elif averaged_elements == 4:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][
                key]) / averaged_elements
        elif averaged_elements == 5:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] +
                          df_data[i + 3][key]) / averaged_elements
        elif averaged_elements == 6:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] +
                          df_data[i + 3][key] +
                          df_data[i + 4][key]) / averaged_elements
        elif averaged_elements == 7:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] +
                          df_data[i + 3][key] +
                          df_data[i + 4][key] + df_data[i + 5][key]) / averaged_elements
        elif averaged_elements == 8:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] +
                          df_data[i + 3][key] +
                          df_data[i + 4][key] + df_data[i + 5][key] + df_data[i + 6][key]) / averaged_elements
        elif averaged_elements == 9:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] +
                          df_data[i + 3][key] +
                          df_data[i + 4][key] + df_data[i + 5][key] + df_data[i + 6][key] + df_data[i + 7][key]) / averaged_elements
        elif averaged_elements == 10:
            return SKO * (df_data[i - 1][key] + df_data[i][key] + df_data[i + 1][key] + df_data[i + 2][key] +
                          df_data[i + 3][key] +
                          df_data[i + 4][key] + df_data[i + 5][key] + df_data[i + 6][key] + df_data[i + 7][key] +
                          df_data[i + 8][key]) / averaged_elements

    df_data_new = []

    for k in range(iteration_count):

        if 10 in averaged_elements:
            for i in range(1, len(df_data) - 8):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                   df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class'] == df_data[i + 6]['class'] == \
                   df_data[i + 7]['class'] == df_data[i + 8]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 10, SKO)
                else:
                    continue
        if 9 in averaged_elements:
            for i in range(1, len(df_data) - 7):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                   df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class'] == df_data[i + 6]['class'] == \
                   df_data[i + 7]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 9, SKO)
                else:
                    continue
        if 8 in averaged_elements:
            for i in range(1, len(df_data) - 6):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                   df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class'] == df_data[i + 6]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 8, SKO)
                else:
                    continue
        if 7 in averaged_elements:
            for i in range(1, len(df_data) - 5):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                   df_data[i + 3]['class'] == df_data[i + 4]['class'] == df_data[i + 5]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 7, SKO)
                else:
                    continue
        if 6 in averaged_elements:
            for i in range(1, len(df_data) - 4):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                   df_data[i + 3]['class'] == df_data[i+4]['class']:
                    # SKO = random.randint(low, up) / 1000
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 6, SKO)
                else:
                    continue
        if 5 in averaged_elements:
            for i in range(1, len(df_data) - 3):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class'] == \
                   df_data[i + 3]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 5, SKO)
                else:
                    continue
        if 4 in averaged_elements:
            for i in range(1, len(df_data) - 2):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class'] == df_data[i + 2]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 4, SKO)
                else:
                    continue
        if 3 in averaged_elements:
            for i in range(1, len(df_data)-1):
                if df_data[i]['class'] == df_data[i - 1]['class'] == df_data[i + 1]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 3, SKO)
                else:
                    continue
        if 2 in averaged_elements:
            for i in range(1, len(df_data)):
                if df_data[i]['class'] == df_data[i - 1]['class']:
                    df_data_new = get_hapalick_params(df_data_new, df_data, i, 2, SKO)
                else:
                    continue
        if 1 in averaged_elements:
            for i in range(0, len(df_data)):
                df_data_new = get_hapalick_params(df_data_new, df_data, i, 1, SKO)

    return pd.DataFrame(df_data_new)


def get_classification_report(data_train, data_test, class_names):
    """Function calculates the accuracy of disease recognition by the neural network.

    :param data_train:  Frame of train data.
    :param data_test:   Frame of test data.
    :param class_names: List of disease names.
    :return:            Accuracy of classification.
    """
    x_train = data_train.iloc[:, 0:30]
    y_train = data_train.iloc[:, 30]

    x_test = data_test.iloc[:, 0:30]
    y_test = data_test.iloc[:, 30]

    print('Train size = {}, test size = {}'.format(len(y_train), len(y_test)))

    clf = MLPClassifier(activation='logistic',
                        max_iter=1200,
                        hidden_layer_sizes=(100,),
                        solver='lbfgs',
                        early_stopping=True)

    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print(acc)

    # show confusion matrix
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=class_names))
    disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.close()

    return acc


def get_statistics(accs):
    """Function prints information about max, miv and average accuracy.

    :param accs:        List of accuracies.
    :return:            None.
    """
    print('\nResults')
    print(accs)
    print("MAX Accuracy = {}\nMIN Accuracy = {}\nAVG Accuracy = {}\n  ".format(max(accs), min(accs), mean(accs)))
    plt.hist(accs, bins=10, cumulative=False)
    plt.grid(linewidth=0.2)
    plt.show()


def get_accuracy(class_names, root_folder, stat_count, input_train_params, input_test_params, SKO=0.05, iteration_count=100, averaged_elements=[10], is_generate=True):
    """Function calls the classification process and outputs the best texture parameters to .csv file.

    :param class_names:         List of disease names.
    :param root_folder:         Path to dir with disease folders.
    :param stat_count:          Number of times the classification accuracy is calculated.
    :param input_train_params:  Name of .csv file with textural params for training.
    :param input_test_params:   Name of .csv file with textural params for testing.
    :param SKO:                 The noise value.
    :param iteration_count:     Count of iterations of generating parameter sets from the averaged group.
    :param averaged_elements:   Count of averaging params.
    :return:                    None.
    """
    accs = []

    for i in range(stat_count):
        print("Iteration # {}".format(i + 1))

        with open(input_train_params) as train, open(input_test_params) as test:
            data_train = pd.read_csv(train, sep='\t')
            if is_generate:
                data_train = generate_haralick_params(data_train.T.to_dict(), SKO, iteration_count, averaged_elements)

            data_test = pd.read_csv(test, sep='\t')
        # uncomment to get extra  test images
        # data_test = generate_haralick_params(data_test.T.to_dict(), 0.05, 50, [1])
        accs.append(get_classification_report(data_train, data_test, class_names))

        if accs[len(accs) - 1] == max(accs):
            data_train.to_csv(root_folder + 'best_generated_params.csv', sep='\t', index=False)

    # show statistic results
    if stat_count > 1:
        get_statistics(accs)
