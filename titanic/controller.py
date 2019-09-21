from titanic.model import TitanicModel
import pandas as pd
import numpy as np
from sklearn.svm import SVC

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

        return m.hook_process(t1, t2)


        #print("---------------------- train / test  head & column --------------------------------")
        #print(t1.head())
        #print(t1.columns)
        #print(train.head())
        #print(train.columns)

    def create_model(self):
        train = self._train
        #axis = 1 컬럼
        model = train.drop('Survived', axis=1)
        print('---- Model Info ----')
        print(model.info)
        return model

    def create_dummy(self) -> object:
        train = self._train
        dummy = train['Survived']
        return dummy

    def test_all(self):
        model = self.create_model()
        dummy = self.create_dummy()
        m = self._m
        m.hook_test(model, dummy)

    def submit(self):
        m = self._m
        model = self.create_model()
        dummy = self.create_dummy()
        test = m.test
        test_id = m.test_id

        clf = SVC()
        clf.fit(model, dummy)
        prediction = clf.predict(test)
        print(prediction)
        submission = pd.DataFrame(
            {'PassengerId':test_id, 'Survived': prediction})
        #submission.to_csv(m.context + 'submission.csv', index=False)
        #submission.to_csv('./data/submission.csv', index=False)


