from dados_treinamento import dados1x1
from perceptron import perceptron

def main():
    
    #dados1x1("char que eu quero q ele reconheça", "char que quero testar")
    training_data = dados1x1("G","D")
    
    # Chamada da função de treinamento
    weights, bias, ciclos = perceptron(training_data)
    print("Pesos finais:", weights)
    print("Bias final:", bias)
    print("Ciclos:", ciclos)
    
if __name__ == "__main__":
    main()