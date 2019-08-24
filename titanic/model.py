"""
survival	생존여부	0 = No, 1 = Yes
pclass	    승선권 	1 = 1st, 2 = 2nd, 3 = 3rd
sex	        성별
Age	        나이 in years
sibsp	    동반한 형제 자매 , 배우자(가족의 동승 여부)   # of siblings / spouses aboard the Titanic
parch	    동반한 부모, 자식# of parents / children aboard the Titanic
ticket	    티켓 번호 number
fare	    티켓의 요금 Passenger fare
cabin	    객실번호 number
embarked	승선한 항구명 C = Cherbourg, Q = Queenstown, S = Southampton

Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

변수 = feature = dimension
"""
#pandas는 string
#numpy는 숫자
import pandas as pd
import numpy as np

class TitanicModel:
    def __init__(self):
        self._context = None
        self._fname = None
        #지도학습 비지도학슴을 나누는 기준(train과 test로 나누면 지도학습)
        self._train = None
        self._test = None
        self._test_id = None

    #람다는 cpu의 레지서터를 이용하여 연산, 일반 블락형식은 메모리에 할당하여 연산
    #인스턴스 변수의 객체변수 의미
    #자바와 다르게 파라미터 기준으로 게터, 세터 변수 의미 구분
    #getter

    @property
    def context(self) -> object:return self._context
    #setter
    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> object: return self._fname

    # setter
    @context.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> object: return self._train

    # setter
    @context.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test

    # setter
    @context.setter
    def test(self, test): self._test = test

    @property
    def test_id(self) -> object : return self._test_id

    # setter
    @test_id.setter
    def test_id(self, test_id): self._test_id = test_id

    #str형식으로 반환한다

    def new_file(self) -> str: return self._context + self._fname

    def new_dframe(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

    def hook_process(self, train, test) -> object:
        print("---------------1. cabin Ticket 삭제 ---------------------")
        t = self.drop_feature(train, test, "Cabin")
        t = self.drop_feature(t[0], t[1], "Ticket")
        print("---------------2. embarked 승선한 항구명 norminal 편집---------------------")
        t = self.embarked_norminal(t[0], t[1])
        print("---------------3. Title 편집---------------------")
        t = self.title_norminal(t[0], t[1])
        print("---------------4. Name, PassengerId 삭제---------------------")
        t = self.drop_feature(t[0], t[1], "Name")
        self._test_id = test["PassengerId"]
        t = self.drop_feature(t[0], t[1], "PassengerId")
        print("---------------5. Age 편집---------------------")
        t = self.age_ordinal(t[0], t[1])
        print("---------------6. Fare ordinal 편집---------------------")
        t = self.fare_oridinal(t[0], t[1])
        print("---------------7. Fare 삭제---------------------")
        t = self.drop_feature(t[0], t[1], "Fare")
        print("---------------8. Sex norminal 편집---------------------")
        t = self.sex_norminal(t[0], t[1])
        t[1] = t[1].fillna({"FareBand":1})
        a = self.null_sum(t[1])
        print("널의 수량{} 개".format(a))
        self._test = t[1]
        return t[0]

        return t[0]

    @staticmethod
    def null_sum(train) -> int:
        return train.isnull().sum()



    @staticmethod
    def drop_feature(train, test, feature) -> []:
        #axis -> 축
        train = train.drop([feature], axis = 1)
        test = test.drop([feature], axis = 1)
        return [train, test]

    @staticmethod
    def embarked_norminal(train, test) -> []:
        # c_city = train[train['Embarked'] == 'C'].shape[0]
        # s_city = train[train['Embarked'] == 'S'].shape[0]
        # q_city = train[train['Embarked'] == 'Q'].shape[0]

        #Embrake에 매핑되는 것이 없으면 소수로 표시 {"Embarked" : "s"}) 소문자 s로 하면 매핑되는 값 이 없어서 null 그대로 변환
        #자료형에 변환 영향을 준다
        train = train.fillna({"Embarked" : "S"})
        city_mapping = {"S" : 1, "C" : 2, "Q" : 3}
        train["Embarked"] = train["Embarked"].map(city_mapping)
        test["Embarked"] = test["Embarked"].map(city_mapping)
        #print(train.head())
        #print(train.columns)
        return [train, test]

    #상관관계에 대한 판단
    @staticmethod
    def title_norminal(train, test) -> []:
        combine = [train, test]
        #[A-Za-z]+\. -> 한글자 의미
        for dataset in combine:
            dataset["Title"] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

        for dataset in combine:
            #\ -> 한줄로 인식하게 해주는 예약어 // 들여쓰기 무시
            dataset["Title"] \
                = dataset["Title"].replace(["Capt", "Col", "Don", "Dr", "Major", "Rev", "Jonkheer", "Dona"], 'Rare')
            dataset["Title"] \
                = dataset["Title"].replace(["Countess", "Lady", "Sir"], 'Royal')
            dataset["Title"] \
                = dataset["Title"].replace(["Mile", "Ms"], 'Miss')
         #   dataset["Title"] \
           #     = dataset["Title"].replace(["Mne", "Mrs"], 'Mrs')
        train[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()
        #print(train[["Title", "Survived"]].groupby(["Title"], as_index=False).mean())
        title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Royal":5, "Rare":6, "Mne":7}
        for dataset in combine:
            dataset["Title"] = dataset["Title"].map(title_mapping)
            #na(null의미)의 경우 0으로 채워라
            dataset["Title"] = dataset["Title"].fillna(0)
        return [train, test]

    @staticmethod
    def sex_norminal(train, test) -> []:
        combine = [train, test]
        sex_mapping = {'male':0, 'female':1}
        for dataset in combine:
            dataset["Sex"] = dataset["Sex"].map(sex_mapping)

        return [train, test]

    @staticmethod
    def age_ordinal(train, test):
        train["Age"] = train["Age"].fillna(-0.5)
        test["Age"] = test["Age"].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ["unknown", "Baby", "Child", "Teenager", "Student", "Young Adult", "Adult", "Senior"]
        train["AgeGroup"] = pd.cut(train["Age"], bins, labels=labels)
        test["AgeGroup"] = pd.cut(train["Age"], bins, labels=labels)
        age_title_mapping = {0: "unknown", 1:"Baby", 2:"Child", 3:"Teenager", 4:"Student", 5:"Young Adult", 6:"Adult", 7:"Senior"}

        for x in range(len(train["AgeGroup"])):
            if train["AgeGroup"][x] == "Unknown":
                train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        for x in range(len(test["AgeGroup"])):
            if test["AgeGroup"][x] == "Unknown":
                test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

        age_mapping = {"unknown":0, "Baby":1, "Child":2, "Teenager":3,"Student":4,"Young Adult":5,"Adult":6,"Senior":7}

        train["AgeGroup"] = train["AgeGroup"].map(age_mapping)
        test["AgeGroup"] = train["AgeGroup"].map(age_mapping)
        print(train["AgeGroup"].head())
        return [train, test]

    @staticmethod
    def fare_oridinal(train, test) -> []:
        #요금단위를 자동으로 4분의1로 구분
        train["FareBand"] = pd.qcut(train['Fare'], 4, labels={1, 2, 3, 4})
        test["FareBand"] = pd.qcut(train['Fare'], 4, labels={1, 2, 3, 4})
        return [train, test]


