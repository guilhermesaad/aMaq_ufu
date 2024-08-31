def perceptron(training_data, max_epochs=100):
    # Inicialização de pesos e bias
    weights = [0] * len(training_data[0][0])  # Inicializa todos os pesos como 0
    bias = 0  # Inicializa o bias como 0
    alpha = 1  # Taxa de aprendizado

    # Treinamento do Perceptron
    for epoch in range(max_epochs): 
        weights_changed = False
        
        # Iteração sobre cada par de treinamento s:t (s = lista com entradas, t = objetivo)
        for (s, t) in training_data:
            # Definição das ativações de entrada
            x = s
            # Cálculo da resposta da unidade de saída
            y_in = bias + sum(x_i * w_i for x_i, w_i in zip(x, weights))
            y = activation(y_in)

            # Atualização dos pesos e bias se houver erro
            if y != t:
                weights = [w_i + alpha * t * x_i for w_i, x_i in zip(weights, x)]
                bias += alpha * t
                weights_changed = True

        # Verificação da condição de parada
        if not weights_changed:
            break  # Se nenhum peso mudou, o treinamento para
      
    return weights, bias, epoch
    
# Função de ativação
def activation(y_in, theta=0): #definido para theta = 0
    if y_in >= theta:
        return 1 
    else:
        return -1
