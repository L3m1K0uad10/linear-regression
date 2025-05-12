import numpy as np



"""  
y = a x + b
y = a x + a x + ... + a x + b
     1 1   2 2         n n
"""

class LinearRegression:
    """  
    data (np.array): complete data along with the dependent variable
    n (int): number of rows
    index (int): dependent variable column index
    NB: index is not the position but the index
    """
    def __init__(self, data, index):
        self._data = data 
        self._index = index
        self._rows, self._cols = data.shape

        self.__b = 0 # intercept
        self.__a = np.zeros((self._cols - 1, 1)) # minus 1 because of one col represents the dependent variable

        self.__sum_X = np.zeros((self._cols, 1))
        self.__sum_Y = np.zeros((self._cols, 1))
        self.__sum_XY = np.zeros((self._cols, 1)) 
        self.__sum_X2 = np.zeros((self._cols, 1)) # sum(X^2)
        self.__sum_X_2 = np.zeros((self._cols, 1)) # sum(X)^2

        self._compute_summations()

    def _compute_summations(self):
        """  
        compute sum(X), sum(Y), sum(XY), sum(X^2), sum(X)^2
        there are np.array of 2 dim where each row represent a sum of a X_i
        """
        # sum_X, sum_X2 and sum_XY
        Y = self._data[0:self._rows + 1, self._index:self._index + 1]
        for i in range(self._cols):
            sum_ = 0
            sum__ = 0
            sum___ = 0
            index = i
            for j, value in enumerate(self._data[0:self._rows + 1, i:i + 1]):
                sum_ += value 
                sum__ += value ** 2
                sum___ += value * Y[j]
        
            self.__sum_X[index] = sum_
            self.__sum_X2[index] = sum__
            self.__sum_XY[index] = sum___
        
        # sum_Y
        self.__sum_Y = self.__sum_X[self._index]

        # sum_X_2
        for i, sum_ in enumerate(self.__sum_X):
            self.__sum_X_2[i] = sum_ * sum_

        self.__sum_X = np.delete(self.__sum_X, index, axis = 0)
        self.__sum_XY = np.delete(self.__sum_XY, index, axis = 0)
        self.__sum_X2 = np.delete(self.__sum_X2, index, axis = 0)
        self.__sum_X_2 = np.delete(self.__sum_X_2, index, axis = 0)
        

        """ print(self.__sum_Y)
        print("\n")
        print(self.__sum_X)
        print("\n")
        print(self.__sum_XY)
        print("\n")
        print(self.__sum_X2)
        print("\n")
        print(self.__sum_X_2) """


    def _compute_intercept(self):
        if self._cols > 2:
            # it is a multiple linear regression
            self.__b = 1 
        else:
            # it is a simple linear regression
            self.__b = ((self.__sum_Y[0] * self.__sum_X2[0]) - (self.__sum_X[0] * self.__sum_XY[0])) / ((self._rows * self.__sum_X2[0]) - self.__sum_X_2[0])

    def _compute_slope(self):
        for i in range(self._cols - 1):
            a = ((self._rows * self.__sum_XY[i]) - (self.__sum_X[i] * self.__sum_Y[0])) / ((self._rows * self.__sum_X2[i]) - self.__sum_X_2[i])
            self.__a[i] = a

    def predict(self, y):
        """
        y: dependent variable
        """  
        self._compute_slope()
        self._compute_intercept()
        print(self.__a)
        print(self.__b)



l = LinearRegression(np.array([[3, 2, 2], [8, 10, 1]]), 2)
l.predict(12)

""" data = np.array([[3, 8],
                 [9, 6],
                 [5, 4],
                 [3, 2]])
l = LinearRegression(data, 1)
l.predict(12) """
    