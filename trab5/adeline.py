import numpy as np
import matplotlib.pyplot as plt
from dados import training_inputs, targets  # Importa a base de dados

class Adaline:
    def __init__(self, n_inputs, learning_rate=0.01, tolerance=1e-3, max_epochs=1000):
        self.weights = np.random.uniform(-0.5, 0.5, n_inputs)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs

    def predict(self, inputs):
        # Calcula o valor líquido das entradas
        return np.dot(inputs, self.weights) + self.bias

    def train(self, training_inputs, targets):
        errors = []  # Lista para armazenar o erro quadrático total em cada época
        for epoch in range(self.max_epochs):
            total_error = 0.0
            max_weight_change = 0.0
            for inputs, target in zip(training_inputs, targets):
                y_liquido = self.predict(inputs)
                
                # Calcula o erro
                error = target - y_liquido
                total_error += error**2  # Acumula o erro quadrático
                
                # Atualiza os pesos
                delta_w = self.learning_rate * error * inputs
                self.weights += delta_w
                self.bias += self.learning_rate * error

                # Calcula a maior mudança de peso
                max_weight_change = max(max_weight_change, np.max(np.abs(delta_w)))
            
            errors.append(total_error / len(training_inputs))  # Erro quadrático médio

            # Verifica a condição de parada
            if max_weight_change < self.tolerance:
                print(f"Treinamento concluído após {epoch+1} épocas.")
                break
        else:
            print("Treinamento finalizado sem convergência.")

        # Plota o erro quadrático total ao longo das épocas
        plt.plot(errors)
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Médio')
        plt.title('Erro Quadrático durante o Treinamento do Adaline')
        plt.show()

# Função para calcular os coeficientes a e b
def calcular_coeficientes(training_inputs, targets):
    x = training_inputs[:, 0]
    y = targets
    n = len(x)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    a = np.mean(y) - b * np.mean(x)
    
    return a, b

# Função para calcular o coeficiente de correlação de Pearson
def calcular_pearson(training_inputs, targets):
    x = training_inputs[:, 0]
    y = targets
    n = len(x)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    
    r = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    return r, r**2

# Função para plotar a linha de regressão
def plotar_regressao(a, b, training_inputs, targets):
    plt.scatter(training_inputs[:, 0], targets, color='blue', label='Dados')
    plt.plot(training_inputs[:, 0], a + b * training_inputs[:, 0], color='red', label='Linha de Regressão')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Treinar o Adaline e traçar a linha de regressão
adaline = Adaline(n_inputs=2, learning_rate=0.01, tolerance=1e-3, max_epochs=1000)
adaline.train(training_inputs, targets)

# Coeficientes da regressão
a, b = calcular_coeficientes(training_inputs, targets)
print(f"Coeficientes da regressão: a = {a:.4f}, b = {b:.4f}")

# Coeficiente de correlação de Pearson e coeficiente de determinação
r, r2 = calcular_pearson(training_inputs, targets)
print(f"Coeficiente de correlação de Pearson: r = {r:.4f}")
print(f"Coeficiente de determinação: r² = {r2:.4f}")

# Plotar a linha de regressão
plotar_regressao(a, b, training_inputs, targets)


# Exemplo de uso
if __name__ == "__main__":
    # Inicializa o modelo Adaline
    adaline = Adaline(n_inputs=2, learning_rate=0.01, tolerance=1e-3, max_epochs=1000)
    
    # Treina o modelo
    adaline.train(training_inputs, targets)

    # Testa o modelo
    for inputs in training_inputs:
        print(f"Entrada: {inputs} -> Saída prevista: {adaline.predict(inputs):.2f}")
