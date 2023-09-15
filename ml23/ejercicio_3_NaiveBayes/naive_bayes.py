import numpy as np

class NaiveBayes():
    def __init__(self, alpha=1) -> None:
        self.alpha = 1e-10 if alpha < 1e-10 else alpha

    def fit(self, X, y):
        # TODO: Calcula la probabilidad de que una muestra sea positiva P(y=1)
<<<<<<< HEAD
        self.prior_positives = np.mean(y) 

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        self.prior_negative = 1- self.prior_positives
=======
        # self.prior_positives = 

        # TODO: Calcula la probabilidad de que una muestra sea negativa P(y=0)
        # self.prior_negative = 
>>>>>>> upstream/master

        # TODO: Para cada palabra del vocabulario x_i
        # calcula la probabilidad de: P(x_i| y=1)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_positives = [P(x_1| y=1), P(x_2| y=1), ..., P(x_n| y=1)]
<<<<<<< HEAD
        self._likelihoods_positives = np.array([np.mean(X[y == 1][:, i]) for i in range(X.shape[1])])
=======
        # self._likelihoods_positives = 
>>>>>>> upstream/master
        
        # TODO:  Para cada palabra del vocabulario x_i, calcula P(x_i| y=0)
        # Guardalas en un arreglo de numpy:
        # self._likelihoods_negatives = [P(x_1| y=0), P(x_2| y=0), ..., P(x_n| y=0)]

<<<<<<< HEAD
        self._likelihoods_negatives = np.array([np.mean(X[y == 0][:, i]) for i in range(X.shape[1])])
=======
        # self._likelihoods_negatives = _likelihoods_negatives
>>>>>>> upstream/master
        return self

    def predict(self, X):
        # TODO: Calcula la distribución posterior para la clase 1 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 1 | X) = P(y=1) * P(x1|y=1) * P(x2|y=1) * ... * P(xn|y=1)
<<<<<<< HEAD
        posterior_positive = self.prior_positives * np.prod(self._likelihoods_positives * X + (1 - self._likelihoods_positives) * (1 - X), axis=1)
=======
        # posterior_positive = 
>>>>>>> upstream/master

        # TODO: Calcula la distribución posterior para la clase 0 dado los nuevos puntos X
        # utilizando el prior y los likelihoods calculados anteriormente
        # P(y = 0 | X) = P(y=0) * P(x1|y=0) * P(x2|y=0) * ... * P(xn|y=0)
<<<<<<< HEAD
        posterior_negative = self.prior_negative * np.prod(self._likelihoods_negatives * X + (1 - self._likelihoods_negatives) * (1 - X), axis=1)

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        clase = 1 if posterior_positive > posterior_negative else 0
        return clase
=======
        # posterior_negative = 

        # TODO: Determina a que clase pertenece la muestra X dado las distribuciones posteriores
        # clase = 
        return
>>>>>>> upstream/master
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)