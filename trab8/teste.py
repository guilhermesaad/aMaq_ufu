import numpy as np
import matplotlib.pyplot as plt  # Importar matplotlib para o gráfico
from digitos import fontes

# -- FUNÇÃO PRINCIPAL --
def main():
    # Inserir uma nova matriz 5x5 para ser testada
    matriz_teste = [
        [0, 0, 1, 0, 0],  
        [0, 0, 1, 0, 0],
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
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    
    saidas = saidas * 5  # Repetir para cada fonte
    saidas = np.array(saidas)
    
    # Parâmetros
    neuronios_entrada = len(entrada[0])
    neuronios_escondidos = 100
    neuronios_saida = 10
    alpha = 0.005
    erro_total_admissivel = 0.0001
    epocas = 10000

    # Treinamento da rede
    print(" -- Treinamento da MLP --")
    v, bv, w, bw, erro_quadratico_total = treinar_rede_multicamadas(entrada, saidas, alpha, epocas, erro_total_admissivel, neuronios_entrada, neuronios_escondidos, neuronios_saida)

    # Plotar o gráfico do erro quadrático total
    plt.plot(erro_quadratico_total)
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Total')
    plt.title('Erro durante o Treinamento da MLP')
    plt.grid(True)
    plt.show()

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
    resultado = testar_rede([matriz_teste_flat], v, bv, w, bw)
    print(f"Previsão para a matriz de teste: {resultado[0]}")

# Função para treinar a rede multicamadas
def treinar_rede_multicamadas(x, t, alpha, epocas, erro_total_admissivel, neuronios_entrada, neuronios_escondidos, neuronios_saida):
    # Inicialização dos pesos
    v = np.random.rand(neuronios_entrada, neuronios_escondidos) - 0.5
    bv = np.random.rand(neuronios_escondidos) - 0.5
    w = np.random.rand(neuronios_escondidos, neuronios_saida) - 0.5
    bw = np.random.rand(neuronios_saida) - 0.5

    erro_quadratico_total = np.zeros(epocas)
    
    for epoca in range(epocas):
        erro_total = 0
        
        for padroes in range(len(x)):
            #  Fase de Feedforward
            z_in = np.dot(x[padroes], v) + bv
            z = bipolar_sigmoid(z_in)
            
            y_in = np.dot(z, w) + bw
            y = bipolar_sigmoid(y_in)
            
            #  Retropropagação do erro 
            d_k = (t[padroes] - y) * bipolar_sigmoid_derivative(y)
            D_w = alpha * np.outer(z, d_k)
            D_bw = alpha * d_k
            
            d_v = np.dot(d_k, w.T) * bipolar_sigmoid_derivative(z)
            D_v = alpha * np.outer(x[padroes], d_v)
            D_bv = alpha * d_v
            
            # Atualização dos pesos
            w += D_w
            bw += D_bw
            v += D_v
            bv += D_bv
            
            erro_total += 0.5 * np.sum((t[padroes] - y) ** 2)
        
        erro_quadratico_total[epoca] = erro_total
        
        if erro_total < erro_total_admissivel:
            print(f"Treinamento concluído na época {epoca + 1} com erro total {erro_total}.")
            break
        
        if epoca % 1000 == 0:
            print(f"Época {epoca + 1}: Erro = {erro_total}")
    
    return v, bv, w, bw, erro_quadratico_total

# Função de ativação bipolar sigmoid
def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

# Derivada da função de ativação bipolar sigmoid
def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

# Função de teste
def testar_rede(X_teste, v, bv, w, bw):
    z_in = np.dot(X_teste, v) + bv  # Camada oculta
    z = bipolar_sigmoid(z_in)
    
    y_in = np.dot(z, w) + bw  # Camada de saída
    y = bipolar_sigmoid(y_in)
    
    previsoes = np.argmax(y, axis=1)  # Escolher o índice da maior probabilidade
    return previsoes

# Chamada da função principal
if __name__ == "__main__":
    main()
