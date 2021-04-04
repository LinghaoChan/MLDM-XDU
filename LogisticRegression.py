import numpy as np


class LogisticRegression(object):
    def __init__(self, method="gradientdescent", eps = 1e-5):

        # initialization
        self.beta = beta = np.ones((1, 3))
        self.method = method
        self.eps = eps

    def fit(self, X, y):
        if self.method == "gradientdescent":
            return self.gradient(X, y)
        elif self.method == "newtown":
            return self.newton(X, y)

    def sigmoid(self, x):
        """
        Sigmoid function.
        Input:
            x:np.array
        Return:
            y: the same shape with x
        """

        y = 1.0 / (1 + np.exp(-x))
        return y


    def gradient(self, X, y, lr=0.5):
        """
        Input:
            X: np.array with shape [N, d]. Input.
            y: np.array with shape [N, 1]. Label.
        Return:
            beta: np.array with shape [1, d+1]. Optimal params with gradient descent method
        """

        N = X.shape[0]
        ones = np.ones((N, 1))

        # padding with one vector
        X1 = np.hstack([X, ones])

        #shape [N, 1]
        z = X1.dot(self.beta.T)

        # log-likehood
        old_l = 0
        new_l = np.sum(-y*z + np.log(1+np.exp(z)))  # 计算对数似然的代价函数值
        iters = 0
        while(np.abs(old_l - new_l) > self.eps):
            #shape [N, 1]
            p1 = np.exp(z) / (1 + np.exp(z))

            #shape [N, N]
            p = np.diag((p1 * (1-p1)).reshape(N))

            #shape [1, d]
            first_order = -np.sum(X1 * (y - p1), 0, keepdims=True)

            # update
            self.beta -= lr * first_order
            z = X1.dot(self.beta.T)
            old_l = new_l
            new_l = np.sum(-y*z + np.log(1+np.exp(z)))

            iters += 1
        print("梯度下降法收敛的迭代次数iters: ", iters)
        print('梯度下降法收敛后对应的代价函数值: ', new_l)
        return self.beta

    def newton(self, X, y):
        """
        Input:
            X: np.array with shape [N, d]. Input.
            y: np.array with shape [N, 1]. Label.
        Return:
            beta: np.array with shape [1, d+1]. Optimal params with newton method
        """
        N = X.shape[0]
        ones = np.ones((N, 1))
        X1 = np.hstack([X, ones])

        #shape [N, 1]
        z = X1.dot(self.beta.T)

        # log-likehood
        old_l = 0
        new_l = np.sum(-y*z + np.log(1+np.exp(z)))  # 计算对数似然的代价函数值
        iters = 0
        while(np.abs(old_l - new_l) > self.eps):
            #shape [N, 1]
            p1 = np.exp(z) / (1 + np.exp(z))

            #shape [N, N]
            p = np.diag((p1 * (1-p1)).reshape(N))

            #shape [1, d+1]
            first_order = -np.sum(X1 * (y - p1), 0, keepdims=True)

            #shape [d+1, d+1]
            second_order = X1.T .dot(p).dot(X1)

            # update
            self.beta -= first_order.dot(np.linalg.inv(second_order))
            z = X1.dot(self.beta.T)
            old_l = new_l
            new_l = np.sum(-y*z + np.log(1+np.exp(z)))

            iters += 1
        print("牛顿法收敛的迭代次数iters: ", iters)
        print('牛顿法收敛后对应的代价函数值: ', new_l)
        return self.beta
