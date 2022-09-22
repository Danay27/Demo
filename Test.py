from openpyxl import Workbook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import HS

if __name__ == '__main__':
    vardim = 10
    function = 'Sphere'
    a = []
    b = []
    for i in range(1):
        bound = np.tile([[-5.12], [5.12]], vardim,)
        hs1 = HS.HarmonySearch(10, vardim, bound, 7000, [0.9, 0.99, 0.008], function)
        hs1.solve()
        a.append(hs1.trace[:, 0])

        bound = np.tile([[-5.12], [5.12]], vardim)
        hs2 = HS.HarmonySearch(2, vardim, bound, 5000, [0.9, 0.99, 0.008], function)
        hs2.solve()
        b.append(hs2.trace[:, 0])

    #x = np.arrange(0, 10000)
    print(f"hs-10 MEAN:{np.mean(a)}, STD:{np.std(a)}")   #mean求取均值,std求取标准差，是方差的平方根
    print(f"hs-2 MEAN:{np.mean(b)}, STD:{np.std(b)}")
    # y1 = hs1.trace[:, 0]
    # y2 = hs2.trace[:, 0]
    # mybook1 = Workbook()
    # wa1 = mybook1.active
    # wa1.append(hs1.trace[:, 0])
    # mybook1.save('hs-10.xlsx')
    #
    # mybook2 = Workbook()
    # wa2 = mybook2.active
    # wa2.append(hs2.trace[:, 0])
    # mybook2.save('hs-2.xlsx')
    #
    # y1 = pd.read_excel('hs-10.xlsx')
    # y2 = pd.read_excel('hs-2.xlsx')
    np.save("hs-10.npy", hs1.trace[:, 0])
    np.save("hs-2.npy", hs2.trace[:, 0])
    y1 = np.load("hs-10.npy")
    y2 = np.load("hs-2.npy")
    # plt.plot(y1, 'r', label='HS-10 best value', marker='D', markevery=250, linewidth=1.0)
    # plt.plot(y2, 'b', label='HS-2 best value', marker='o', markevery=250, linewidth=1.0)
    plt.plot(y1, 'k', label='HS-10 best value', marker='D', markevery=250, linewidth=1.0)
    plt.plot(y2, 'k', label='HS-2 best value', marker='o', markevery=250, linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("function value")
    plt.title("Harmony search algorithm for function optimization")
    plt.legend()
    plt.show()



