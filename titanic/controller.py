from titanic.model import TitanicModel
import pandas as pd
import numpy as np

class TitanicController:
    def __init__(self):
        self._m = TitanicModel()
        self._context = './data/'
        self._train = self.create_train()

    def create_train(self) -> object:
        m = self._m
        m.context = self._context
        m.fname = "train.csv"
        t1 = m.new_dframe()
        m.fname = "test.csv"
        t2 = m.new_dframe()

        train = m.hook_process(t1, t2)
        #print("---------------------- train / test  head & column --------------------------------")
        #print(t1.head())
        #print(t1.columns)
        #print(train.head())
        #print(train.columns)


