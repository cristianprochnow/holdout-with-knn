import numpy as np
from ucimlrepo import fetch_ucirepo

class HoldoutKnn:
    __dataset = {}

    __features = []
    __targets = []

    K_LIMIT = 10

    def fetch_breast_cancer_wisconsin(self):
        self.__dataset = fetch_ucirepo(id=17)

    def fetch_iris(self):
        self.__dataset = fetch_ucirepo(id=53)

    def __fetch_data(self):
        self.__features = self.__dataset.data.features.to_numpy()
        self.__targets = self.__dataset.data.targets.to_numpy().flatten()

    def __serialize_data(self):
        if self.__targets.dtype == "O":
            unique_class_values = np.unique(self.__targets)
            class_values_association = {target: idx for idx, target in enumerate(unique_class_values)}
            self.__targets = np.array([class_values_association[target] for target in self.__targets])

    def __euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def __shuffle_results(self):
        indexes = np.arange(len(self.__features))

        np.random.shuffle(indexes)

        self.__features, self.__targets = self.__features[indexes], self.__targets[indexes]

    def __check_proportion(self, train_size, value_size, test_size):
        if not np.isclose(train_size + value_size + test_size, 1.0):
            raise ValueError(
                "The sum of train_size, value_size and test_size must be equal to 1."
            )
        
    def __separate_dataset(self, train_size, value_size, test_size):
        n = len(self.__features)
        n_train = int(n * train_size)
        n_val = int(n * value_size)
        n_test = int(n * test_size)

        feature_train, feature_value, feature_test = (
            self.__features[:n_train],
            self.__features[n_train : n_train + n_val],
            self.__features[n_train + n_val : n_train + n_val + n_test],
        )
        target_train, target_value, target_test = (
            self.__targets[:n_train],
            self.__targets[n_train : n_train + n_val],
            self.__targets[n_train + n_val : n_train + n_val + n_test],
        )

        return feature_train, feature_value, feature_test, target_train, target_value, target_test
    
    def __simple_holdout(self, train_size=0.7, value_size=0.15, test_size=0.15):
        self.__check_proportion(train_size, value_size, test_size)
        self.__shuffle_results()
        return self.__separate_dataset(train_size, value_size, test_size)
    
    def __knn(self, feature_train, target_train, feature_test, K):
        predictions = []

        for test in feature_test:
            distances = [self.__euclidean_distance(test, train) for train in feature_train]
            k_indexes = np.argsort(distances)[:K]
            k_labels = target_train[k_indexes]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)

        return np.array(predictions)
        
    def run(self):
        self.__fetch_data()
        self.__serialize_data()
    
        feature_train, feature_value, feature_test, target_train, target_value, target_test = self.__simple_holdout(
            train_size=0.7, value_size=0.15, test_size=0.15
        )

        best_accuracy = 0
        best_K = 1

        for K in range(1, self.K_LIMIT):
            target_prediction_value = self.__knn(feature_train, target_train, feature_value, K)
            accuracy = np.mean(target_prediction_value == target_value)

            print(f"k = {K}, Acurácia na validação: {accuracy:.2f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_K = K
        
        print(f"k = {K}, Acurácia na validação: {accuracy:.2f}")

        target_prediction_test = self.__knn(feature_train, target_train, feature_test, best_K)
        accuracy_test = np.mean(target_prediction_test == target_test)

        print(f"Acurácia no teste: {accuracy_test:.2f}")

holdout_knn = HoldoutKnn()

holdout_knn.fetch_breast_cancer_wisconsin()
# holdout_knn.fetch_iris()
holdout_knn.run()