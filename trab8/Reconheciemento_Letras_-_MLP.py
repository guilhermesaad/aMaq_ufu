import numpy as np
from digitos import fontes

# -- FUNÇÃO PRINCIPAL --
def main():
    # Inserir uma nova matriz 5x5 para ser testada
    matriz_teste = [
        [1, 1, 1, 1, 0],  
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]
    
    # Configuração dos dados
    entrada = fontes
    
    saidas = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ]
    saidas = saidas * 5  # Repetir 5 vezes para cada fonte
    saidas = np.array(saidas)
    
    # Parâmetros
    neuronios_entrada = len(entrada[0])
    neuronios_escondidos = 50
    neuronios_saida = 10
    alpha = 0.05
    erro_total_admissivel = 0.001
    epocas = 10000

    # Inicializar pesos
    pesos_escondidos = np.random.randn(neuronios_entrada, neuronios_escondidos) * 0.01
    pesos_saida = np.random.randn(neuronios_escondidos, neuronios_saida) * 0.01

    # Treinamento da rede
    print(" -- Treinamento da MLP --")

    pesos_escondidos, pesos_saida = treinar_rede(entrada, saidas, pesos_escondidos, pesos_saida, alpha, erro_total_admissivel, epocas)

    print("\n -- Teste da MLP --")

    print("Matriz de teste:")
    for linha in matriz_teste:
        for elemento in linha:
            if elemento == 1:
                print("#", end=" ")
            else:
                print(".", end=" ")
        print()

    # Achatar a matriz para a entrada
    matriz_teste_flat = np.array(matriz_teste).flatten()

    # Testar a nova matriz usando os pesos treinados
    resultado = testar_rede([matriz_teste_flat], pesos_escondidos, pesos_saida)
    print(f"Previsão para a matriz de teste: {resultado[0]}")

def treinar_rede(entrada, saidas, pesos_escondidos, pesos_saida, alpha, erro_total_admissivel, epocas):
    for epoca in range(epocas):
        # Feedfoward
        z_escondido = bipolar_sigmoid(np.dot(entrada, pesos_escondidos))  # Camada oculta
        z_saida = softmax(np.dot(z_escondido, pesos_saida))  # Camada de saída
        
        # Calcular o erro 
        erro = calcular_erro(z_saida, saidas)

        # Retropropagação
        erro_saida = z_saida - saidas  # Derivada da função de erro
        erro_escondido = np.dot(erro_saida, pesos_saida.T) * bipolar_sigmoid_derivative(z_escondido)  
        
        # Atualizar pesos
        pesos_saida -= alpha * np.dot(z_escondido.T, erro_saida)
        pesos_escondidos -= alpha * np.dot(entrada.T, erro_escondido)
        
        # Mostrar progresso a cada 100 épocas
        if epoca % 100 == 0:
            print(f"Época {epoca}: Erro = {erro}")

        # Condição para parar o treinamento se o erro for menor que o admissível
        if erro < erro_total_admissivel:
            print(f"Treinamento concluído na época {epoca} com erro {erro}.")
            break
    
    return pesos_escondidos, pesos_saida

# Função de ativação bipolar sigmoid
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivada da função de ativação bipolar sigmoid
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

# Função de teste
def testar_rede(x_teste, pesos_escondidos, pesos_saida):
    z_escondido = bipolar_sigmoid(np.dot(x_teste, pesos_escondidos))          # Camada oculta
    z_saida_teste = softmax(np.dot(z_escondido, pesos_saida))                 # Camada de saída
    
    # Vetor de possibilidades previstas
    previsoes = np.argmax(z_saida_teste, axis=1)
    return previsoes

# -- FUNÇÕES PARA ENONTRAR POSSIBILIDADES --
# Função softmax para calcular as probabilidades
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Função de cálculo do erro (entropia cruzada)
def calcular_erro(y, t):
    return -np.sum(t * np.log(y + 1e-8))  # Adicionando 1e-8 para evitar log(0)

# Chamada da função principal
if __name__ == "__main__":
    main()
