import numpy as np



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

        self.__sum_X = np.zeros((self._cols - 1, )) 
        self.__sum_Y = 0 
        self.__sum_X2 = np.zeros((self._cols - 1, )) # x^2
        self.__sum_XiXj = np.zeros((self._cols - 1, self._cols - 1))
        self.__sum_XY = np.zeros((self._cols - 1, ))  

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

    def _solve_equations(self):
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
        left_side_equations = np.zeros((self._cols, self._cols))
        right_side_equations = np.zeros((self._cols, ))
        
        left_side_equations[0] = np.append(self.__sum_X, self._rows)
        right_side_equations[0] = self.__sum_Y[0]
        
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
            left_side_equations[i + 1] = np.append(symmetric_matrix[i], self.__sum_X[i])
            right_side_equations[i + 1] = self.__sum_XY[i]
        
       
        # solving for [a1, a2,..., an]
        coefficients, residuals, rank, s = np.linalg.lstsq(left_side_equations, right_side_equations, rcond=None)
        
        return coefficients
    
    def linear_model(self):
        coefficients = self._solve_equations()

        return coefficients

    def predict(self, x:np.array)->float:
        """
        x np.array: independent(s) variable(s)
        considering that x array is ordered as it should
        """ 
        try:
            if x.shape == (self._cols - 1, ):
                coefficients = self._solve_equations()
                sum_ = 0
                slope = coefficients[self._cols - 1]
                for i in range(self._cols - 1):
                    sum_ += coefficients[i] * x[i]
                result = sum_ + slope

                return result
            else:
                raise TypeError("parameter error: unexpected shape, unmatched shape")
        except Exception as e:
            print(f"An error occurred: {e}")


if "__main__" == __name__:
    """
    # TEST ONEP: SUCCESS
    data = np.array([
        [140, 60, 22],
        [155, 62, 25],
        [159, 67, 24],
        [179, 70, 20],
        [192, 71, 15],
        [200, 72, 14],
        [212, 75, 14],
        [215, 78, 11]
    ])
    l = LinearRegression(data, 0)
    l.predict(np.array([3, 2]))
    # should get
    # b = -6.867
    # a1 = 3.148
    # a2 = -1.656
    """

    # TEST TWO: SUCCESS
    data = np.array([
        [5, 40],
        [7, 120],
        [12, 180],
        [16, 210],
        [20, 240]
    ])

    l = LinearRegression(data, 1)
    summary = l.linear_model()
    print(summary)
    result = l.predict(np.array([5]))
    print(result)
    # should get
    # b = 11.506
    # a = 12.208


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