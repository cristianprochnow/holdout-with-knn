# Holdout simples com KNN

# Equipe

- Cristian Prochnow
- Gustavo Henrique Dias
- Lucas Willian de Souza Serpa
- Marlon de Souza
- Ryan Gabriel Mazzei Bromati

# Sobre

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Para esse algoritmo criamos uma classe com o processamento necessário, para demonstrar de uma forma mais clara os passos envolvidos em todo o passo a passo. Então, para começar, temos que instanciar essa classe.

```python
holdout_knn = HoldoutKnn()
```

Então começamos com a busca da base de dados diretamente com a *lib* da UCI, relacionada com a busca de *datasets* específicas do site.

```python
holdout_knn.fetch_breast_cancer_wisconsin()
# holdout_knn.fetch_iris()
```

Assim, podemos usar o mesmo algoritmo para analisar diversas bases de dados diferentes, desde que o *dataset* esteja presente na UCI Machine Learning Repository.

Próximo passo é a execução do algoritmo, ao qual todo o processamento está contido. Para isso, podemos chamar a função abaixo da classe que criamos.

```python
holdout_knn.run()
```

O primeiro passo dessa função é buscar e organizar os dados, que são feitos por meio dos métodos abaixo.

```python
self.__fetch_data()
self.__serialize_data()
```

A primeira função é responsável apenas por pegar os dados da estrutura retornada pela biblioteca do repositório de *datasets*. Já, a segunda função, fica responsável por organizar os dados, caso necessário. Isto é, as funções usadas no processo não aceitam classes de formato aquém de `Float64`, então esse método traduz os dados presentes para o formato esperado.

```python
feature_train,
feature_value, 
feature_test, 
target_train, 
target_value, 
target_test = self.__simple_holdout(
    train_size=0.7, value_size=0.15, test_size=0.15
)
```

Esse próximo passo é a separação dos dados para serem usados na análise. Com isso, usamos a separação para **treinamento**, **avaliação** e **teste**, usando a proporção de 70%, 15% e 15%.

```python
def __simple_holdout(self, train_size=0.7, value_size=0.15, test_size=0.15):
    self.__check_proportion(train_size, value_size, test_size)
    self.__shuffle_results()
    return self.__separate_dataset(train_size, value_size, test_size)
```

Dentro dessa separação, começamos pela parte de verificar os valores que foram passados nos parâmetros, ou seja, se a soma de `train_size`, `value_size` e `test_size` for diferente de  `1`, retorna erro. A `__shuffle_results` fica responsável por embaralhar os resultados antes de separá-los, para que sempre tenhamos uma abordagem diferente para cada vez que o algoritmo for executado. E, para finalizar, a `__separate_dataset` realiza as separações dos dados, usando os delimitadores de `slice` que o Python possibilita usar (índices juntamente com o operador de `:`).

```python
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
```

E, então, após preparar os dados, o processamento de predição é executado. O treinamento então é realizado em `K_LIMIT` vezes, conforme o valor que for determinado nessa constante. Para cada rodada, a função `__knn` é chamada, ao qual implementará o algoritmo de KNN que estamos abordando nesse contexto. Após essas `K_LIMIT` rodadas, o último passo é executado, que então pega o valor de vizinhos `K` que trouxe a melhor precisão, e então executa novamente o processo usando como base os dados separados para teste.

```python
def __knn(self, feature_train, target_train, feature_test, K):
	  predictions = []
	
	  for test in feature_test:
	      distances = [self.__euclidean_distance(test, train) for train in feature_train]
	      k_indexes = np.argsort(distances)[:K]
	      k_labels = target_train[k_indexes]
	      prediction = np.bincount(k_labels).argmax()
	      predictions.append(prediction)
	
	  return np.array(predictions)
```

O algoritmo de KNN usa então o passo a passo de percorrer os resultados que estão sendo usados como teste, treinando assim a base maior.

O primeiro passo realiza os cálculos de distãncia do resultado teste em questão, com todo o restante de resultados que temos que determinar para treinamento. Com o cálculo da `__euclidean_distance` a distância euclidiana é determinada dos resultados paralelos.

Na próxima linha então o `argsort` organiza os resultados obtidos no cálculo euclidiano, organizando da menor distância até a maior. Então, com o `:K` pegamos somente os vizinhos determinado no parâmetro.

Na próxima linha pegamos apenas os vizinhos — do treinamento — que estão mais pertos do elemento atual da base de teste. Para, na próxima, executar então a contagem de elementos (com a `bincount`) conforme a classe que cada uma das linhas selecionadas carrega, para então pegar o valor de classe que mais apareceu nessa organização foi feita (`argmax`).

E, para finalizar, a última linha do *loop* pega o valor escolhido e coloca na lista de valores escolhidos conforme a votação, para então a linha de retorno da função transformar essa lista no formato NumPy.

