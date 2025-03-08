"""
Cristian Prochnow
Gustavo Henrique Dias
Lucas Willian de Souza Serpa
Marlon de Souza
Ryan Gabriel Mazzei Bromati

https://github.com/guusdias/KNN
"""

import numpy as np
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=17)
X = iris.data.features.to_numpy()
y = iris.data.targets.to_numpy().flatten()

print("Tipo dos rótulos (y):", y.dtype)

if y.dtype == "O":
    classes_unicas = np.unique(y)
    mapa_classes = {classe: idx for idx, classe in enumerate(classes_unicas)}
    y = np.array([mapa_classes[classe] for classe in y])


def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn(X_train, y_train, X_test, k):
    predicoes = []
    for x in X_test:
        distancias = [distancia_euclidiana(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distancias)[:k]
        k_labels = y_train[k_indices]
        predicao = np.bincount(k_labels).argmax()
        predicoes.append(predicao)
    return np.array(predicoes)


def holdout_simples(X, y, train_size=0.7, val_size=0.15, test_size=0.15):

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            "A soma de train_size, val_size e test_size deve ser igual a 1."
        )

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    n = len(X)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    n_test = int(n * test_size)

    X_train, X_val, X_test = (
        X[:n_train],
        X[n_train : n_train + n_val],
        X[n_train + n_val : n_train + n_val + n_test],
    )
    y_train, y_val, y_test = (
        y[:n_train],
        y[n_train : n_train + n_val],
        y[n_train + n_val : n_train + n_val + n_test],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = holdout_simples(
    X, y, train_size=0.7, val_size=0.15, test_size=0.15
)

melhor_acuracia = 0
melhor_k = 1

for k in range(1, 10):
    y_pred_val = knn(X_train, y_train, X_val, k)
    acuracia = np.mean(y_pred_val == y_val)
    print(f"k = {k}, Acurácia na validação: {acuracia:.2f}")

    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_k = k

print(f"Melhor k: {melhor_k} com acurácia de {melhor_acuracia:.2f} na validação")

y_pred_test = knn(X_train, y_train, X_test, melhor_k)
acuracia_test = np.mean(y_pred_test == y_test)
print(f"Acurácia no teste: {acuracia_test:.2f}")
