# -*-coding:utf-8-*-

"""
牛顿法实现对率回归(Logistic Regression)
来源: '机器学习, 周志华'
模型: P69 problem 3.3
数据集: P89 watermelon_3.0a (watermelon_3.0a.npy)
"""
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import LogisticRegression as LogisReg


if __name__ == "__main__":

    # read data from xlsx file
    workbook = xlrd.open_workbook("3.0alpha.xlsx")
    sheet = workbook.sheet_by_name("Sheet1")
    X1 = np.array(sheet.row_values(0))
    X2 = np.array(sheet.row_values(1))

    # this is the extension of x
    X3 = np.array(sheet.row_values(2))
    y = np.array(sheet.row_values(3))
    X = np.vstack([X1, X2]).T
    y = y.reshape(-1, 1)

    # plot training data
    for i in range(X1.shape[0]):
        if y[i, 0] == 0:
            plt.plot(X1[i], X2[i], 'r+')

        else:
            plt.plot(X1[i], X2[i], 'bo')




    # get optimal parameters beta with gradient descent method
    method1 = "gradientdescent"
    optmizer1 = LogisReg.LogisticRegression(method=method1)
    beta1 = optmizer1.fit(X, y)
    newton_left1 = -(beta1[0, 0]*0.1 + beta1[0, 2]) / beta1[0, 1]
    newton_right1 = -(beta1[0, 0]*0.9 + beta1[0, 2]) / beta1[0, 1]

    # 显示最终的结果
    plt.plot([0.1, 0.9], [newton_left1, newton_right1], 'g-', label=method1)

    # get optimal parameters beta with newton method
    method2 = "newtown"
    optmizer2 = LogisReg.LogisticRegression(method=method2, eps=1e-2)
    beta2 = optmizer2.fit(X, y)
    newton_left2 = -(beta2[0, 0]*0.1 + beta2[0, 2]) / beta2[0, 1]
    newton_right2 = -(beta2[0, 0]*0.9 + beta2[0, 2]) / beta2[0, 1]

    # 显示最终的结果
    plt.plot([0.1, 0.9], [newton_left2, newton_right2], 'r-', label=method2)


    plt.legend()
    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title("Logistic Regression")
    plt.show()
