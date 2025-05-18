import numpy as np



"""  
y = a x + b
y = a x + a x + ... + a x + b
     1 1   2 2         n n

∑y= a1∑x1 + a2∑x2 + a3∑x3 + a4∑x4 + nb

∑x1y= a1∑x1^2 + a2∑x1x2 + a3∑x1x3 + a4∑x1x4 + b∑x1
∑x2y= a1∑x1x2 + a2∑x2^2 + a3∑x2x3 + a4∑x2x4 + b∑x2
∑x3y= a1∑x1x3 + a2∑x2x3 + a3∑x3^2 + a4∑x3x4 + b∑x3
∑x4y= a1∑x1x4 + a2∑x2x4 + a3∑x3x4 + a4∑x4^2 + b∑x4

b = Y' - a1X1' - a2X2'
Y' is mean of Y
X1 is mean of X1
X2 is mean of X2
TODO: - now solve the equations with numpy linear algebra
        returns a1, a2, a3, a4, b coefficients
        [a1, a2, a3, a4, b] ≈ [1.0, 2.0, 0.5, 0.0, 3.0]
      - composed the final regression equation
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

        self.__sum_X = np.zeros((self._cols - 1, )) 
        self.__sum_Y = 0 
        self.__sum_X2 = np.zeros((self._cols - 1, )) # x^2
        self.__sum_XiXj = np.zeros((self._cols - 1, self._cols - 1))
        self.__sum_XY = np.zeros((self._cols - 1, ))  

        self.__sum_X_2 = np.zeros((self._cols, 1)) #- (X)^2


        self._compute_summations()

    def _compute_summations(self):
        """  
        compute sum(X), sum(Y), sum(XY), sum(X^2), sum(X)^2
        there are np.array of 2 dim where each row represent a sum of a X_i
        """
        # sum_X, sum_X2 and sum_XY
        Y = self._data[:, self._index]
        data = np.delete(self._data, self._index, axis = 1) # data without the dependant variable column
        
        # sum_X, sum_X2 and sum_XY
        for i in range(self._cols - 1):
            sum_ = 0
            sum__ = 0
            for j, value in enumerate(data[:, i]):
                sum_ += value 
                sum__ += value ** 2
            XiY = data[:, i] * Y
            self.__sum_X[i] = sum_
            self.__sum_X2[i] = sum__ 
            self.__sum_XY[i] = np.sum(XiY)

        # sum_Y 
        self.__sum_Y = np.array([np.sum(Y)])

        # sum_XiXj 
        # it's going to follow a left-aligned inverted right-angled triangle shape
        # as np does not support inhomogeneous shape will fill with dummy 0's
        for i in range(self._cols - 1):
            for j in range(i+1, self._cols - 1, 1):
                XiXj = data[:, i] * data[:, j]
                sum_ = np.sum(XiXj) 
                self.__sum_XiXj[i][j - (i+1)] = sum_

    """ def _compute_intercept(self):
        if self._cols > 2:
            # it is a multiple linear regression
            self.__b = ((1 / self._rows) * self.__sum_Y[0])
            second_terms = 0
            for i, a in enumerate(self.__a):
                second_terms -= a * (1 / self._rows) * self.__sum_X[i]
            self.__b += second_terms
        else:
            # it is a simple linear regression
            self.__b = ((self.__sum_Y[0] * self.__sum_X2[0]) - (self.__sum_X[0] * self.__sum_XY[0])) / ((self._rows * self.__sum_X2[0]) - self.__sum_X_2[0])

    def _compute_slope(self):
        # ∑x1y= a1∑x1^2 + a2∑x1x2 + a3∑x1x3 + a4∑x1x4 + b∑x1
        for i in range(self._cols - 1):
            a = ((self._rows * self.__sum_XY[i]) - (self.__sum_X[i] * self.__sum_Y[0])) / ((self._rows * self.__sum_X2[i]) - self.__sum_X_2[i])
            self.__a[i] = a """

    def _equations(self):
        """ 
        ∑y= a1∑x1 + a2∑x2 + a3∑x3 + a4∑x4 + nb

        ∑x1y= a1∑x1^2 + a2∑x1x2 + a3∑x1x3 + a4∑x1x4 + b∑x1
        ∑x2y= a1∑x1x2 + a2∑x2^2 + a3∑x2x3 + a4∑x2x4 + b∑x2
        ∑x3y= a1∑x1x3 + a2∑x2x3 + a3∑x3^2 + a4∑x3x4 + b∑x3
        ∑x4y= a1∑x1x4 + a2∑x2x4 + a3∑x3x4 + a4∑x4^2 + b∑x4
        

        equations = |    ∑x1    ∑x2    ∑x3    ∑x4    n   |
                    |   ∑x1^2  ∑x1x2  ∑x1x3  ∑x1x4  ∑x1  |
                    |   ∑x1X2  ∑x2^2  ∑x2x3  ∑x2x4  ∑x2  |
                    |   ∑x1x3  ∑x2x3  ∑x3^2  ∑x3x4  ∑x3  |
                    |   ∑x1x4  ∑x2x4  ∑x3x4  ∑x4^2  ∑x4  |

        """
        equations = np.zeros((self._cols, self._cols))
        
        equations[0] = np.append(self.__sum_X, self._rows)
        
        # rearrangement: shift the arrays to the end
        k = 1 # track the number of time of each shift to the right
        for i in range(self._cols - 1 - 1):# the second -1, because last one is only zero's
            for j in range((self._cols - 1) - 1 - k, -1, -1):
                self.__sum_XiXj[i][j + k] = self.__sum_XiXj[i][j]
                self.__sum_XiXj[i][j] = 0.
            k += 1
        
        # fill diagonal
        np.fill_diagonal(self.__sum_XiXj, self.__sum_X2)
        
        # Mirror upper triangle to lower triangle
        symmetric_matrix = self.__sum_XiXj + self.__sum_XiXj.T - np.diag(self.__sum_X2)
        
        for i in range(self._cols - 1):
            equations[i + 1] = np.append(symmetric_matrix[i], self.__sum_X[i])
        print(equations)

    def predict(self, x:np.array)->float:
        """
        x np.array: independent(s) variable(s)
        """ 
        try:
            if x.shape == (1, self._cols - 1):
                self._equations()
                """ self._compute_slope()
                self._compute_intercept()
                #print(self.__a)
                #print(self.__b) """
            else:
                raise TypeError("parameter error: unexpected shape, unmatched shape")
        except Exception as e:
            print(f"An error occurred: {e}")


""" l = LinearRegression(np.array([[3, 2, 2], [8, 10, 1]]), 2)
l.predict(np.array([[3, 2]])) """

""" data = np.array([[3, 8],
                 [9, 6],
                 [5, 4],
                 [3, 2]])
l = LinearRegression(data, 1)
l.predict(12) """

data = np.array([[10, 1, 2, 3, 4], 
                 [8, 2, 1, 0, 3],
                 [9, 0, 3, 1, 2]])

l = LinearRegression(data, 0)
l.predict(np.array([[3, 2, 0, 1]]))
# should get
# b = -6.867
# a1 = 3.148
# a2 = -1.656
    
""" 
[[X0],
 [X1],
 ...,
 [Xi]]

[[X0X1, X0X2,..., X0Xn], 
 [X1X1, X1X2,..., X1Xn],
 ...,
 [XiX1, XiX2,..., XiXn]]

X1      X2      X3      X4      Y
1       2       3       4       10
2       1       0       3       8
0       3       1       2       9
= 3     = 6     = 4     = 9     = 27
-------------------------------------
X1^2      X2^2      X3^2      X4^2     
1         4         9         16       
4         1         0         9      
0         9         1         4       
= 5       = 14      = 10      = 29   
-------------------------------------
X1X2       X1X3       X1X4
1 x 2 = 2  1 x 3 = 3  1 x 4 = 4
2 x 1 = 2  2 x 0 = 0  2 x 3 = 6
0 x 1 = 0  0 x 1 = 0  0 x 2 = 0
= 4        = 3        = 10

X2X3       X2X4       
2 x 3 = 6  2 x 4 = 8  
1 x 0 = 0  1 x 3 = 3  
3 x 1 = 3  3 x 2 = 6  
= 9        = 17        

X3X4              
3 x 4 = 12  
0 x 3 = 0  
1 x 2 = 2  
= 14
-----------------------------------------------
X1Y          X2Y          X3Y          X4Y
1 x 10 = 10  2 x 10 = 20  3 x 10 = 30  4 x 10 = 40
2 x  8 = 16  1 x  8 = 8   0 x  8 = 0   3 x  8 = 24
0 x  9 = 0   3 x  9 = 27  1 x  9 = 9   2 x  9 = 18
= 26         = 55         = 39         = 82

"""