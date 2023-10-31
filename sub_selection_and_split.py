import numpy as np
import pandas as pd
from submodlib import FacilityLocationFunction, DisparitySumFunction, DisparityMinFunction, LogDeterminantFunction, \
    FeatureBasedFunction, GraphCutFunction, SetCoverFunction
from submodlib_cpp import FeatureBased
from scipy.spatial import distance
import os
import shutil
import time
import random


def select_subset_and_split(algo: str, budget: int, ground_data: pd.DataFrame(), optimizer='NaiveGreedy') -> None:
    supported_algos = ['FacilityLocationFunction', 'DisparitySumFunction', 'DisparityMinFunction',
                       'LogDeterminantFunction', 'FeatureBasedFunction',
                       'GraphCutFunction', 'SetCoverFunction', 'Random']

    assert algo in supported_algos, 'Algo not supported'

    start_time = time.time()

    groundData = ground_data

    groundData_transpose = groundData.transpose()

    num_images = len(groundData_transpose)

    print(f'Starting {algo} algo ...')
    if algo == 'FacilityLocationFunction':
        objFL = FacilityLocationFunction(n=num_images, data=np.array(groundData_transpose), separate_rep=False,
                                         mode="dense", metric="euclidean")
        greedyList = objFL.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                    stopIfNegativeGain=False, verbose=False)
    elif algo == 'FeatureBasedFunction':
        distanceMatrix = distance.cdist(np.array(groundData_transpose), np.array(groundData_transpose),
                                        metric="euclidean")
        similarityMatrix = 1 - distanceMatrix
        features = []
        for i in range(num_images):
            features.append(similarityMatrix[i].tolist())
        objFB = FeatureBasedFunction(n=num_images, features=features, numFeatures=len(features), sparse=False,
                                     mode=FeatureBased.logarithmic)
        greedyList = objFB.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                    stopIfNegativeGain=False, verbose=False)
    elif algo == 'GraphCutFunction':
        lambda_value = 0.3
        objGC = GraphCutFunction(n=num_images, mode="dense", separate_rep=False, lambdaVal=lambda_value,
                                 data=np.array(groundData_transpose), metric="euclidean")
        greedyList = objGC.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                    stopIfNegativeGain=False, verbose=False)
    elif algo == 'SetCoverFunction':
        num_concepts = num_images
        num_samples = num_images
        cover_set = []
        np.random.seed(1)
        random.seed(1)
        # concept_weights = np.random.rand(num_concepts).tolist()
        # printables = []
        for i in range(num_samples):
            cover_set.append(set(random.sample(list(range(num_concepts)), random.randint(0, num_concepts / 2))))
            # printable = ["\u25A1"] * num_concepts
            # print(''.join(map(str, temp)))
            # for ind, val in enumerate(printable):
            #     if ind in cover_set[i]:
            #         printable[ind] = "\u25A0"
            # print(i, ": ", ''.join(map(str, printable)))
            # printables.append(printable)
            # printable = ["\u25A0" if index in cover_set[i] for index, val in enumerate(temp)]
        obj = SetCoverFunction(n=num_images, cover_set=cover_set, num_concepts=num_concepts)
        greedyList = obj.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                  stopIfNegativeGain=False, verbose=False)
    elif algo == 'DisparitySumFunction':
        objFL = DisparitySumFunction(n=num_images, data=np.array(groundData_transpose), mode="dense",
                                     metric="euclidean")
        greedyList = objFL.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                    stopIfNegativeGain=False, verbose=False)
    elif algo == 'DisparityMinFunction':
        objFL = DisparityMinFunction(n=num_images, data=np.array(groundData_transpose), mode="dense",
                                     metric="euclidean")
        greedyList = objFL.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                    stopIfNegativeGain=False, verbose=False)
    elif algo == 'LogDeterminantFunction':
        lambda_value = 1
        objFL = LogDeterminantFunction(n=num_images, data=np.array(groundData_transpose), mode="dense",
                                       metric="euclidean", lambdaVal=lambda_value)
        greedyList = objFL.maximize(budget=budget, optimizer=optimizer, stopIfZeroGain=False,
                                    stopIfNegativeGain=False, verbose=False)

    # greedys = [[grounds[i][x[0]] for x in greedyList] for i in range(num_features)]

    if algo != 'Random':
        selected_image_idxs = [greedyList[i][0] for i in range(len(greedyList))]
        selected_image_names = [groundData_transpose.index[i] for i in selected_image_idxs]
    else:
        selected_image_names = random.sample(list(groundData.columns), budget)

    # Deleting existing folder to avoid overwriting of files of or copying files on top of initial files
    try:
        shutil.rmtree('to_train')
    except FileNotFoundError:
        pass

    # Splitting data to folders
    try:
        os.mkdir(f'to_train')
    except FileExistsError:
        pass

    try:
        os.mkdir(f'to_train/0')
    except FileExistsError:
        pass

    try:
        os.mkdir(f'to_train/1')
    except FileExistsError:
        pass

    for i in range(len(selected_image_names)):

        # print(selected_image_names[i])

        img_file_source = os.path.join('lake', selected_image_names[i])

        if selected_image_names[i].split('.')[0][0] == '0':
            img_file_destination = os.path.join(f'to_train', '0', selected_image_names[i])
        elif selected_image_names[i].split('.')[0][0] == '1':
            img_file_destination = os.path.join(f'to_train', '1', selected_image_names[i])
        else:
            print('Image file destination not defined')
            img_file_destination = None

        # print(img_file_source)
        # print(img_file_destination)

        shutil.copy(img_file_source, img_file_destination)

    end_time = time.time()
    print(f'Time taken for computation - {np.round((end_time - start_time) / 60, 2)} mins')
