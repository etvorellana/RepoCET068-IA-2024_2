import numpy as np
import matplotlib.pyplot as plt


def pLagrange(x, x_i, y_i):
    n = len(x_i)
    L = np.zeros(n)
    for i in range(n):
        L[i] = np.prod([(x - x_i[j])/(x_i[i] - x_i[j]) for j in range(n) if i != j])
    return np.dot(y_i, L)

def heuristic(y_predict):
    x = np.array([1.0, 3.0, 5.0, 8.0])
    y = np.array([2.0, 3.0, 1.0, 4.0])
    y_predict = np.array(y_predict)
    
    x_inter = np.linspace(0, 10, len(y_predict))
    y_inter = [pLagrange(xi, x, y) for xi in x_inter]
    y_inter = np.array(y_inter)

    # Calculate the mean squared error (MSE) by taking the mean of the squared differences
    mse = ((y_predict - y_inter) ** 2).mean()

    # Calculate the RMSE by taking the square root of the MSE
    rmse = np.sqrt(mse)
    return rmse    

def bBoxEx(w):
    ref = 18465
    x = [1.0, 3.0, 5.0, 8.0]
    y = [2.0, 3.0, 1.0, 4.0]
    x_extra = [-5.0, 11.0, -4.0, 12.0, -3.0, 13.0, -2.0, 14.0, -1.0, 15.0, 0.0, 16.0]
    y_extra = [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 5.0, 9.0]
    if w <= 0 or w >= 2**16:
        y_inter = np.zeros(101)
    else:
        xlist = []
        ylist = []
        while w > 0:
            iw = w % 2
            ir = ref % 2
            if ir != 0 and iw != 0:
                xlist.append(x.pop())
                ylist.append(y.pop())
            elif(ir != 0 and iw == 0):
                x.pop()
                y.pop()
            elif(iw != 0):
                xlist.append(x_extra.pop())
                ylist.append(y_extra.pop())
            else:
                x_extra.pop()
                y_extra.pop()
            w = w // 2
            ref = ref // 2
        x_inter = np.linspace(0, 10, 101)
        y_inter = [pLagrange(xi, xlist, ylist) for xi in x_inter]

    return x_inter, y_inter      

def bin2list(w):
    l = []
    for i in range(16):
        d = w % 2
        l = [d] + l 
        w = w // 2
    return l


def list2bin(l):
    w = 0
    for i in range(16):
        w = w * 2 + l[i]
    return w




if __name__  == '__main__':
    # Dados
    x = np.array([1.0, 3.0, 5.0, 8.0])
    y = np.array([2.0, 3.0, 1.0, 4.0])
    

    x_inter = np.linspace(0, 10, 101)
    y_inter = [pLagrange(xi, x, y) for xi in x_inter]

    # Gráfico
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(x_inter, y_inter, label='Interpolação de Lagrange')
    ax[0,0].scatter(x, y, label='Pontos Conhecidos', color='red')

    x_infer, y_infer = bBoxEx(18465)
    ax[0,1].plot(x_infer, y_infer, label='Interpolação de Lagrange')
    ax[0,1].scatter(x, y, label='Pontos Conhecidos', color='red')
    print(heuristic(y_infer))

    x_infer, y_infer = bBoxEx(465)
    ax[1,0].plot(x_infer, y_infer, label='Interpolação de Lagrange')
    ax[1,0].scatter(x, y, label='Pontos Conhecidos', color='red')
    print(heuristic(y_infer))

    x_infer, y_infer = bBoxEx(185)
    ax[1,1].plot(x_infer, y_infer, label='Interpolação de Lagrange')
    ax[1,1].scatter(x, y, label='Pontos Conhecidos', color='red')
    print(heuristic(y_infer))

    plt.show()

    rota = 18465
    l = bin2list(rota)
    print(l)
    w = list2bin(l)
    print(w)

    

    
    # Teste
