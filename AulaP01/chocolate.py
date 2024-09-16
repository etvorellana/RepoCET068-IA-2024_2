from random import randint  # Importa a função randint do módulo random

def printBarra(barra):
    for linha in barra:
        print(linha)

def main():
    print("Jogo da barra de chocolate:")
    print("Você e seu amigo estão dividindo uma barra de chocolate.")
    print("Voces decidem fazer uma brincadeira com as seguintes regras.")
    print("1. Um jogador escolhe um tablete específico da barra de chocolate e,")
    print("fica com todos os tabletes que estejam a direita e assima do")
    print("tablete selecionado. " )
    print("2. P próximo jogador escolhe outro tablete dos que restaram e fica.")
    print("com todos os tabletes que estejam a direita e acima do tablete selecionado.")
    print("3. Os jogadores se alternam até que todos os tabletes tenham sido escolhidos.")
    print("4. O jogador que ficar com o último tablete, quelem localizado na parte inferior")
    print(" esquerda da barra de chocolate, perde o jogo.")

    print("Vamos jogar?")

    # Inicializa a barra de chocolate
    barra = []
    col = randint(3, 7) # Número de colunas da barra de chocolate
    lin = randint(5, 9) # Número de linhas da barra de chocolate
    if col == lin:
        col += 1 # para a barra não ficar quadrada
    
    for i in range(lin):
        barra.append(["[]"]*col)
    
    printBarra(barra)

    

main()
        