import numpy as np

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def a_fun(m, n):
    x = mcol(np.arange(m))
    y = mrow(np.arange(n))
    M = x*y
    return np.float64(M)
    
def b_fun(M):
    arySum = M.sum(axis=0)
    return M / mrow(arySum) # Reshaping is not needed
    
def c_fun(M):
    arySum = M.sum(axis=1)
    return M / mcol(arySum) # Reshaping is necessary
    
def d_fun(ary):
    ary = ary.copy()
    indexMask = ary < 0
    ary[indexMask] = 0
    return ary

def e_fun(A, B):
    C = np.dot(A, B)
    return C.sum()

if __name__ == '__main__':

    print ('a.')
    print (a_fun(3, 4))

    M = np.array([[1.0, 2.0, 6.0, 4.0],
                  [3.0, 4.0, 3.0, 7.0],
                  [1.0, 4.0, 6.0, 9.0]])
    
    print ('\nb.')
    print (b_fun(M))

    M = M.T
    print ('\nc.')
    print (c_fun(M))

    M = np.array([[-1.0, 2.0, 3.0],
                  [2.0, -3.0, -4.0]]) # With 2D arrays
    
    print ('\nd.')
    print (d_fun(M))

    x = np.array([1.0, -1.0, 2.0, 3.0, -2.0, 4.0, -7.0]) # With 1D arrays
    print (d_fun(x))
    
    print ('\ne.')
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    B = np.array([[1.0, 3.0],
                  [2.0, 1.0]])
    print (e_fun(A, B))