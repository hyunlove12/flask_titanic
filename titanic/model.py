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
    @context.setter
    def test_id(self, test_id): self._test_id = test_id

    #str형식으로 반환한다

    def new_file(self) -> str: return self._context + self._fname

    def new_dframe(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

    def hook_process(self, train, test) -> object:
        print("---------------1. ---------------------")
        print("---------------2. ---------------------")
        print("---------------3. ---------------------")



