import numpy as np
import glob
import datetime
import math
import random
import os
import shutil


ROOT_PATH = './my_datasets/'

#music_gerne_selected = ['rock','R&B','funk','bossanova','Medium_swing']
music_gerne_selected = ['bossanova','country','funk','rock','shuffle']





for music_gerne in music_gerne_selected:

    # seperate the train npy
    if not os.path.exists(os.path.join(ROOT_PATH,  music_gerne + '/train')):
        os.makedirs(os.path.join(ROOT_PATH, music_gerne + '/train'))
    x = np.load(os.path.join(ROOT_PATH,
                             music_gerne+'_train.npy'))
    print(x.shape)
    count = 0
    for i in range(x.shape[0]):
        if np.max(x[i]):
            count += 1
            np.save(os.path.join(ROOT_PATH,
                                  music_gerne  +'/train/'   + music_gerne + '_train_{}.npy'.format(
                                     i + 1)), x[i])
            print(x[i].shape)

    #seperate the test npy
    if not os.path.exists(os.path.join(ROOT_PATH,  music_gerne + '/test')):
        os.makedirs(os.path.join(ROOT_PATH, music_gerne + '/test'))
    x = np.load(os.path.join(ROOT_PATH,
                             music_gerne+'_test.npy'))
    print(x.shape)
    count = 0
    for i in range(x.shape[0]):
        if np.max(x[i]):
            count += 1
            np.save(os.path.join(ROOT_PATH,
                                  music_gerne + '/test/'  + music_gerne + '_test_{}.npy'.format(
                                     i + 1)), x[i])
            print(x[i].shape)
