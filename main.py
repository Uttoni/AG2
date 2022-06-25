import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def testarDados(xTest):
    prediction = tree.predict(xTest)
    return prediction

data = pd.read_csv('tic-tac-toe.data', delimiter=',')
numericData = data.replace(['x', 'o', 'b', 'positive', 'negative'], [1, -1, 0, 1, -1])
dataArray = np.array(numericData)

x = dataArray[:, :9]
y = dataArray[:, 9:].reshape(dataArray[:, 9:].size)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
print(f"Tamanho x de treino: {x_train.shape}")
print(f"Tamanho x de teste: {x_test.shape}")
print(f"Tamanho y de treino: {y_train.shape}")
print(f"Tamanho y de teste: {y_test.shape}")

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree_predict = testarDados(x_test)
report = classification_report(y_test, tree_predict)

print()
print(report)

userTestData = [[]]
#x,x,x,o,b,o,b,o,b
for i in range(0, 9):
    userTestData[0].append(input('Entre com uma jogada para o caso de teste: '))

userTestData = np.char.replace(np.array(userTestData), 'x', '1')
userTestData = np.char.replace(np.array(userTestData), 'o', '-1')
userTestData = np.char.replace(np.array(userTestData), 'b', '0')
result = testarDados(userTestData)

print(f"Resultado teste do usuário:{result}")
if(result[0]==1):
    print("X Ganhou!! :)")
else:
    print("X não ganhou :(")
