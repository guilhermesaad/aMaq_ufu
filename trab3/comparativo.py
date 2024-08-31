from dados_treinamento import dados1x1
from perceptron import perceptron
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
file_name = "comparativos.txt"


def compara_tudo(letras):
    resposta = ""
    for i in letras:
        for j in letras:
            weights, bias, ciclos = perceptron(dados1x1(i,j))
            resposta += "Char " + str(i) + " com char " + str(j) + "\nCiclos:"+ str(ciclos) + "\n\n"
            
    
    return resposta


with open(file_name, 'w') as file:
    # Escrever a lista no arquivo
    file.write(compara_tudo(chars))


print(f"Arquivo '{file_name}' gerado com sucesso!")

