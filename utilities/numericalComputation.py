import numpy as np

def BFGSAlgo(f,g,xd,epsilon=1.e-6):
    """
       quasi-Newton method, use BFGS algorithm
       :param f:target function, it's a method having a float parameter.
       :param g:the derivative of the target function, it's a method having a float parameter.
       :param xd: the dimension of the vector
       :param epsilon: precision
       :return: the optimization point
    """
    x=np.random.rand(xd)
    B=np.eye(xd)
    gk=g(x)
    if np.linalg.norm(gk)<epsilon:
        return x
    while True:
        pk = np.linalg.solve(B, -1 * gk)
        #pk=solveEquation(B,-1*gk)

        print(np.linalg.norm(gk))

        #stepLength=calLambdaByArmijoRule(x.reshape(-1,1),f(x),gk.reshape(-1,1),pk.reshape(-1,1),f)
        stepLength=0.01
        xPre=x
        gkPre=gk

        x=x+stepLength*pk
        gk=g(x)
        if np.linalg.norm(gk)<epsilon:
            return x

        #update B
        yk=gk-gkPre
        deltaK=x-xPre
        deltaMatrix=np.matmul(deltaK.reshape(-1,1),deltaK.reshape(1,-1))
        B=B+(np.matmul(yk.reshape(-1,1),yk.reshape(1,-1))/np.sum(yk*deltaK))- \
          (np.matmul(np.matmul(B,deltaMatrix),B)/np.matmul(np.matmul(deltaK.reshape(1,-1),B),deltaK.reshape(-1,1)))


def calLambdaByArmijoRule(xCurr, fCurr, gCurr, pkCurr,f, c=1.e-4, v=0.5):
    """
    refer to https://www.cnblogs.com/xxhbdk/p/11785365.html
    to calculate lambda
    """
    i = 0
    alpha = v ** i
    xNext = xCurr + alpha * pkCurr
    fNext = f(xNext)

    while True:
        if fNext <= fCurr + c * alpha * np.matmul(pkCurr.T, gCurr)[0, 0]: break
        i += 1
        alpha = v ** i
        xNext = xCurr + alpha * pkCurr
        fNext = f(xNext)

    return alpha


#PALU decomposition
def PALU_Factorization(A: np.array):
    U = A.copy()
    P = np.eye(U.shape[0])
    L = np.zeros(U.shape)
    for index in range(U.shape[1]):
        maxIndex = index + np.argmax(U[index:, index])
        # exchange 2 rows
        #print(U[[index, maxIndex], :])
        P[[index, maxIndex], :] = P[[maxIndex, index], :]
        U[[index, maxIndex], :] = U[[maxIndex, index], :]
        L[[index, maxIndex], :] = L[[maxIndex, index], :]
        # eliminate non-zero elements
        for rIndex in range(index + 1, U.shape[0]):
            # try:
            #     assert U[index, index]!=0
            # except:
            #     print(index)
            #     print(U)
            multiFactor = U[rIndex, index] / U[index, index]
            U[rIndex, :] -= U[index, :] * multiFactor
            L[rIndex, index] = multiFactor

        # 给L加上对角线的1
    for i in range(U.shape[0]):
        L[i, i] = 1.0
    return P, L, U


# 使用PA=LU分解解方程
def solveEquation(A: np.array, b: np.array):
    P, L, U = PALU_Factorization(A)
    Pb = np.matmul(P, b.reshape(-1,1))
    # 此时方程为：LUx=Pb
    # 先解Lc=Pb
    c = np.zeros([L.shape[0], 1])
    for i in range(L.shape[0]):
        c[i, 0] = Pb[i, 0]
        for j in range(i):
            c[i, 0] -= L[i, j] * c[j, 0]

    # 再解Ux=c
    x = np.zeros([U.shape[0], 1])
    for i in range(U.shape[0] - 1, -1, -1):
        x[i, 0] = c[i, 0]
        for j in range(U.shape[1] - 1, i, -1):
            x[i, 0] -= U[i, j] * x[j, 0]
        x[i, 0] /= U[i, i]

    return x.reshape(-1,)

if __name__ == '__main__':
    f=lambda x:5*x[0]*x[0]+2*x[1]*x[1]+3*x[0]-10*x[1]+4
    g=lambda x:np.array([10*x[0]+3,4*x[1]-10])
    print(BFGSAlgo(f,g,2))
    #print(solveEquation(np.eye(50),np.array([i for i in range(50)])))