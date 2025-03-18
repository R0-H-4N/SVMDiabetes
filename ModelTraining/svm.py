import numpy as np
class SVM:
    def __init__(self):
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.n = None
        self.fitted = False
        self.C = None
        self.gamma = None
        self.kernelFunction = None
        self.noInvalids = False
        self.tol = None
        self.train_accuracy = None
        self.test_accuracy = None
        self.train_losses = None
        self.test_losses = None
    
    def kernel(self, x1, x2):
        if self.kernelFunction == 'linear':
            return x1.T @ x2
        else:
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def fit(self, X, y, C, tol, kernel, gamma):
        self.X = X
        self.y = y
        self.alpha = np.zeros(X.shape[1])
        self.b = 0
        self.n = X.shape[1]
        self.C = C
        self.fitted = True
        self.kernelFunction = kernel
        self.gamma = gamma
        self.tol = tol
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def predict(self, X):
        result = 0
        for i in range(self.n):
            result += self.alpha[i] * self.y[i] * self.kernel(self.X[:, i], X)
        return result + self.b
    
    def violatesKKT(self, i):
        y = self.y[i]
        alpha = self.alpha[i]
        g = y * self.predict(self.X[:, i]) - 1
        if alpha < 1e-5:
            return g < 1e-5
        elif abs(alpha - self.C) < 1e-5:
            return g > 1e-5
        else:
            return abs(g) > 1e-5
        
    def violatingIndices(self):
        violaters = []
        errorsViolaters = []

        nonViolaters = []
        errorsNonViolaters = []

        for i in range(self.n):
            if self.violatesKKT(i):
                violaters.append(i)
                errorsViolaters.append(self.predict(self.X[:, i]) - self.y[i])
            else:
                nonViolaters.append(i)
                errorsNonViolaters.append(self.predict(self.X[:, i]) - self.y[i])
        
        I, E_i = None, -float('inf')
        J, E_j = None, -float('inf')

        # print(violaters, errorsViolaters)
        # print(nonViolaters, errorsNonViolaters)
        self.noInvalids = len(violaters) == 0
        if len(violaters) == 0:
            A = nonViolaters
            B = errorsNonViolaters
        elif len(violaters) > 1:
            A = violaters
            B = errorsViolaters
        else:
            A = violaters + nonViolaters
            B = errorsViolaters + errorsNonViolaters

        #print(A, B)
        temp = -float('inf')
        for idx, err in list(zip(A, B)):
            if abs(err) > temp:
                I, E_i = idx, err
                temp = abs(err)
        
        temp = -float('inf')
        for idx, err in list(zip(A, B)):
            if I != idx and abs(err - E_i) > temp:
                J, temp = idx, abs(err - E_i)
        
        return I, J
    

    def hingeLoss(self, X, y):
        y_pred = np.array([(self.predict(bob)) for bob in X.T])
        return np.sum(np.maximum(0, 1 - y * y_pred))

    def train(self, X, y, X_test, y_test, C=1, tol=1e-3, maxIter = 1000, kernel = 'linear', gamma = 1):
        if not self.fitted:
            self.fit(X, y, C, tol, kernel, gamma)

        epoches = []


        for _ in range(maxIter):
            #if _ % 10 == 0: print(_)
            i, j = self.violatingIndices()
            if self.y[i] != self.y[j]:  
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:  
                L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])


            K_ii = self.kernel(self.X[:, i], self.X[:, i])
            K_jj = self.kernel(self.X[:, j], self.X[:, j])
            K_ij = self.kernel(self.X[:, i], self.X[:, j])


            eta = 2 * K_ij - K_ii - K_jj
            if eta >= 0:
                continue


            E_i = self.predict(self.X[:, i]) - self.y[i]
            E_j = self.predict(self.X[:, j]) - self.y[j]

            alpha_j_new = self.alpha[j] - self.y[j] * (E_i - E_j) / eta
            alpha_j_new = min(H, max(L, alpha_j_new))

            alpha_i_new = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - alpha_j_new)

            
            b1 = self.b - E_i - self.y[i] * (alpha_i_new - self.alpha[i]) * K_ii - self.y[j] * (alpha_j_new - self.alpha[j]) * K_ij
            b2 = self.b - E_j - self.y[i] * (alpha_i_new - self.alpha[i]) * K_ij - self.y[j] * (alpha_j_new - self.alpha[j]) * K_jj

            if 0 < alpha_i_new < self.C:
                b_new = b1
            elif 0 < alpha_j_new < self.C:
                b_new = b2
            else:
                b_new = (b1 + b2) / 2
            
            if max(abs(alpha_i_new - self.alpha[i]), abs(alpha_j_new - self.alpha[j])) < self.tol and self.noInvalids:
                break
            self.alpha[i] = alpha_i_new
            self.alpha[j] = alpha_j_new
            self.b = b_new

            y_pred = np.array([np.sign(self.predict(bob)) for bob in self.X.T])
            train_accuracy = np.mean(y_pred == self.y)*100
            y_test_pred = np.array([np.sign(self.predict(bob)) for bob in X_test.T])
            test_accuracy = np.mean(y_test_pred == y_test)*100

            self.test_losses.append(self.hingeLoss(X_test, y_test))
            self.train_losses.append(self.hingeLoss(self.X, self.y))

            if _ % 10 == 0: print(f"Iteration: {_+10}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
            epoches.append(_)
            self.test_accuracies.append(test_accuracy)
            self.train_accuracies.append(train_accuracy)
            if train_accuracy > 95:
                break

