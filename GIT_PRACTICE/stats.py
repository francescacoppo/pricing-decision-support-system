def media(lista):
    return sum(lista) / len(lista)

numeri = [10, 20, 30, 40, 50]

risultato = media(numeri)

print(risultato)

def mediana(lista):
    lista_ordinata = sorted(lista)
    n = len(lista_ordinata)
    centro = n // 2
    
    # Se il numero di elementi è dispari
    if n % 2 != 0:
        return lista_ordinata[centro]
    
    # Se il numero di elementi è pari
    else:
        return (lista_ordinata[centro - 1] + lista_ordinata[centro]) / 2

dati = [4, 1, 3, 2]
print(mediana(dati))  