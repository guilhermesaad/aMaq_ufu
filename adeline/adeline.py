import numpy as np
import matplotlib.pyplot as plt
from dados import load_data  # Importa a função de carregamento de dados

class Adaline:
    def __init__(self, n_inputs, learning_rate=0.01, tolerance=1e-3, max_epochs=1000):
        self.weights = np.random.uniform(-0.5, 0.5, n_inputs)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs
        self.errors = []  # Lista para armazenar o erro quadrático total em cada época

    def predict(self, inputs):
        # Calcula o valor líquido das entradas
        return np.dot(inputs, self.weights) + self.bias

    def train(self, training_inputs, targets):
        for epoch in range(self.max_epochs):
            max_weight_change = 0.0
            total_error = 0.0
            for inputs, target in zip(training_inputs, targets):
                y_liquido = self.predict(inputs)
                
                # Calcula o erro
                error = target - y_liquido
                
                # Atualiza os pesos
                delta_w = self.learning_rate * error * inputs
                self.weights += delta_w
                self.bias += self.learning_rate * error

                # Calcula o erro quadrático total
                total_error += error**2

                # Calcula a maior mudança de peso
                max_weight_change = max(max_weight_change, np.max(np.abs(delta_w)))
            
            self.errors.append(total_error)  # Armazena o erro quadrático total da época
            
            # Verifica a condição de parada
            if max_weight_change < self.tolerance:
                print(f"Treinamento concluído após {epoch+1} épocas.")
                break
        else:
            print("Treinamento finalizado sem convergência.")

    def plot_errors(self):
        plt.plot(self.errors)
        plt.title('Erro Quadrático Total durante o Treinamento')
        plt.xlabel('Épocas')
        plt.ylabel('Erro Quadrático Total')
        plt.grid()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Carrega a base de dados
    X, y = load_data()

    # Inicializa o modelo Adaline
    adaline = Adaline(n_inputs=X.shape[1], learning_rate=0.01, tolerance=1e-3, max_epochs=1000)
    
    # Treina o modelo
    adaline.train(X, y)

    # Plota o erro quadrático total
    adaline.plot_errors()
