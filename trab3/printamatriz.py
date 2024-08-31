from dados_treinamento import chars

letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def printa_lista(matriz):
#"▯" ou " " = -1 "▮"=1
    for i in range(len(matriz)):
        if (i)%5 == 0:
            print()
        if matriz[i] == -1:
            print("▯  ", end="")
        else:
            print("▮  ", end="")
    print()

def printa_linha(linha):
    print()
    for i in linha:
        print(i, end="")
        
            

def printa_string(letras):
    for i in letras:
        matriz = chars(i)
        printa_lista(matriz)



