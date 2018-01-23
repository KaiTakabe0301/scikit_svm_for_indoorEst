#!/usr/bin/env python
# coding:UTF-8
import numpy as np
import json
import collections
import os
import tkinter as Tk
import tkinter.filedialog as tkfd
import copy


class WLAN_Positioning:
    def __init__(self):
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []
        self.dimLabel = dict()
        self.testLabel = dict()
        self.di = ''
        self.diTest = ''
        self.diTrain = ''


    def __loadTrain(self):
        trainFiles = os.listdir(self.diTrain)
        testFiles = os.listdir(self.diTest)

        for trainFile in trainFiles:
            tempX = np.loadtxt(self.diTrain + '/' + trainFile, delimiter=',')
            name, ext = os.path.splitext(trainFile)
            for i in range(tempX.shape[0]):
                if name in self.testLabel:  # check key
                    self.train_X.append(tempX[i])
                    self.train_Y.append(self.testLabel[name])

        return [copy.deepcopy(self.train_X),copy.deepcopy(self.train_Y)]

    #同じ座標の測定データが、測定回数分だけ並ぶ構造
    def __loadTest(self):
        testFiles = os.listdir(self.diTest)

        for testFile in testFiles:
            tempX = np.loadtxt(self.diTest + '/' + testFile, delimiter=',')
            name, ext = os.path.splitext(testFile)
            for i in range(tempX.shape[0]):
                if name in self.testLabel:  # check key
                    self.test_X.append(tempX[i])
                    self.test_Y.append(self.testLabel[name])

        return [copy.deepcopy(self.test_X),copy.deepcopy(self.test_Y)]

    def outputZscore(self):
        train_X = np.array(self.train_X)

        # calc of a mean and std of training data's column vector
        mean_X = np.mean(train_X, axis=0)
        std_X = np.std(train_X, axis=0)

        mean_txt="{\n\t\t"
        std_txt="{\n\t\t"
        keys=list(self.dimLabel.keys())
        values=list(self.dimLabel.values())

        for i in range(mean_X.shape[0]):
            print(keys[values.index(i)])

    def loads(self,di='Nothing'):
        if di=='Nothing':
            self.di = tkfd.askdirectory(title="学習セットが保存されているフォルダを指定してください。")
        else:
            self.di = di

        self.diTest = self.di + "/test"
        self.diTrain = self.di + "/train"
        dimPath = open(self.di + '/dimLabel.json', 'r')
        testPath = open(self.di + '/testLabel.json', 'r')

        self.dimLabel = json.load(dimPath, object_pairs_hook=collections.OrderedDict)  # 順番を維持
        self.testLabel = json.load(testPath, object_pairs_hook=collections.OrderedDict)
        dimKeys = self.dimLabel.keys()
        testKeys = self.testLabel.keys()
        train_X, train_Y = self.__loadTrain()
        test_X, test_Y = self.__loadTest()

        return [train_X, train_Y, test_X, test_Y]

    def getDim(self):
        return len(self.dimLabel)

    def getOLayerNum(self):
        return len(self.testLabel)

    def getTestLabel(self):
        return self.testLabel

