import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString


def preprocess_frame(frame):

    # Converte para escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convertendo de volta para BGR para manter a compatibilidade com a função de inferência
    frame_preprocessed = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    
    return frame_preprocessed


def calcular_limites_variaveis(output_filename):
    output_path = os.path.join(output_filename)
    df = pd.read_excel(output_path)

    limites_variaveis = []

    for index, row in df.iterrows():

        limite_variavel = row['proportion'] * 50 # Parâmetro importante

        limites_variaveis.append(limite_variavel)

    return limites_variaveis


def movimento_area(save_folder, output_filename):

    limite = 5000

    output_path = os.path.join(save_folder, output_filename)
    df = pd.read_excel(output_path)

    df['diff'] = df['area'].diff()
    df['diff'] = df['diff'].fillna(0)

    df['movimento'] = abs(df['diff']) > limite

    porcentagem_movimento = df['movimento'].mean() * 100
    
    return porcentagem_movimento


def analisar_movimento(save_folder, output_filename):
    output_path = os.path.join(save_folder, output_filename)
    df = pd.read_excel(output_path)

    limites = calcular_limites_variaveis(output_path)

    df['limite'] = limites

    linha_inicial_ficticia = pd.DataFrame({'tempo': [0], 'corpo_xy': [0], 'limite': [0], 'proportion': [0]})
    df = pd.concat([linha_inicial_ficticia, df], ignore_index=True).sort_values('tempo').reset_index(drop=True)

    # Forçar o valor da primeira linha (agora segunda após a adição) na coluna 'limite' a ser zero
    df.loc[0, 'limite'] = 0

    # Adicionar uma nova linha no final com tempo fictício e valores zero para 'corpo_xy' e 'limite'
    ultimo_tempo = df['tempo'].iloc[-1]
    nova_linha_final = pd.DataFrame({'tempo': [ultimo_tempo + 0.1], 'corpo_xy': [0], 'limite': [0], 'proportion': [0]})
    df = pd.concat([df, nova_linha_final], ignore_index=True).sort_values('tempo').reset_index(drop=True)

    df['movimento'] = df['corpo_xy'] > df['limite']
    df.loc[0, 'movimento'] = None
    df.loc[1, 'movimento'] = None
    df.loc[df.index[-1], 'movimento'] = None

    tempo_movimento = 0
    tempo_total = 30 # df['tempo'].iloc[-1]

    for index in range(1, len(df)):
        diferenca_tempo = df.at[index, 'tempo'] - df.at[index - 1, 'tempo']
        if df.at[index, 'movimento']:
            tempo_movimento += diferenca_tempo
        elif 1.5 < diferenca_tempo <= 4:
            tempo_movimento += diferenca_tempo

    porcentagem_movimento = (tempo_movimento / tempo_total) * 100

    intersections = []

    for i in range(1, len(df)):
        # Verifica se os valores necessários não são nulos
        if pd.isnull(df['tempo'].iloc[i - 1]) or pd.isnull(df['tempo'].iloc[i]) or \
        pd.isnull(df['corpo_xy'].iloc[i - 1]) or pd.isnull(df['corpo_xy'].iloc[i]) or \
        pd.isnull(df['limite'].iloc[i - 1]) or pd.isnull(df['limite'].iloc[i]):
            continue
        
        # Cria as linhas
        line1 = LineString([(df['tempo'].iloc[i - 1], df['corpo_xy'].iloc[i - 1]), 
                            (df['tempo'].iloc[i], df['corpo_xy'].iloc[i])])
        line2 = LineString([(df['tempo'].iloc[i - 1], df['limite'].iloc[i - 1]), 
                            (df['tempo'].iloc[i], df['limite'].iloc[i])])
        
        # Verifica se os pontos das linhas não são idênticos
        if line1.is_empty or line2.is_empty or line1.equals(line2):
            continue
        
        # Calcula a interseção
        intersection = line1.intersection(line2)
        
        # Verifica se a interseção é vazia
        if intersection.is_empty:
            continue
        
        # Adiciona a interseção à lista
        intersections.append((intersection.x, intersection.y))

    # Calcular a área total dos polígonos e imprimir a área de cada intervalo
    total_area = 0
    for i in range(len(intersections) - 1):
        # Verificar se 'corpo_xy' é maior que 'limite' no ponto médio do intervalo
        midpoint_index = (df['tempo'] > intersections[i][0]) & (df['tempo'] < intersections[i + 1][0])
        midpoint = df[midpoint_index].iloc[0]
        if midpoint['corpo_xy'] > midpoint['limite']:
            # Criar polígono para o intervalo atual
            polygon_points = df[(df['tempo'] >= intersections[i][0]) & (df['tempo'] <= intersections[i + 1][0])]
            # Construir os pontos do polígono incluindo os pontos de interseção
            points = [(intersections[i][0], intersections[i][1])] + \
                     polygon_points[['tempo', 'corpo_xy']].values.tolist() + \
                     [(intersections[i + 1][0], intersections[i + 1][1])] + \
                     polygon_points[['tempo', 'limite']].values[::-1].tolist() + \
                     [(intersections[i][0], intersections[i][1])]
            # Verificar se o polígono tem pelo menos 4 coordenadas
            if len(points) < 4:
                print(
                    f"Polígono entre {intersections[i][0]:.2f} e {intersections[i + 1][0]:.2f} não tem coordenadas suficientes. Ignorando intervalo.")
                continue
            polygon = Polygon(points)
            area = polygon.area
            total_area += area
            print(
                f"Entre o intervalo {intersections[i][0]:.2f} e {intersections[i + 1][0]:.2f}, houve uma área de {area:.2f}.")

    # print(df)
    df.to_excel(f"{output_path}_Movimento.xlsx", index=False)

    # print(f"Houve {porcentagem_movimento:.2f}% de inquietude.")

    # print(f"A área total positiva entre as curvas é: {total_area:.2f}")

    # Plotar as curvas
    # plt.plot(df['tempo'], df['corpo_xy'], label='corpo_xy')
    # plt.plot(df['tempo'], df['limite'], label='limite')
    # plt.legend()
    # plt.show()

    # if porcentagem_movimento <= 20 or total_area <= 250:
    #     print("Não está inquieto.")
    # else:
    #     print("Está inquieto.")

    return porcentagem_movimento, total_area


def detectar_rotacao(save_folder, output_filename, window_size, threshold):

    output_path = os.path.join(save_folder, output_filename)
    df = pd.read_excel(output_path)

    rotations = []

    tempo_total = 30  # df['tempo'].iloc[-1] - df['tempo'].iloc[0]

    tempo_rotacao = 0

    interval_start = 0

    for interval_end in range(window_size, len(df['tempo']), window_size):
        window_width = df['width'][interval_start:interval_end]

        mean_width = np.mean(window_width)
        std_dev_width = np.std(window_width)

        if std_dev_width > threshold * mean_width:
            rotations.append(
                f"No intervalo {df['tempo'].iloc[interval_start]} a {df['tempo'].iloc[interval_end]}, houve rotação")
            tempo_rotacao += df['tempo'].iloc[interval_end] - df['tempo'].iloc[interval_start]
        else:
            rotations.append(
                f"No intervalo {df['tempo'].iloc[interval_start]} a {df['tempo'].iloc[interval_end]}, não houve rotação")

        interval_start = interval_end

    # Verificar se há algum restante / NAO É NECESSARIO PARA TAMANHO MULTIPLO
    if interval_start < len(df['tempo']):
        window_width = df['width'][interval_start:]
        mean_width = np.mean(window_width)
        std_dev_width = np.std(window_width)
        if std_dev_width > threshold * mean_width:
            rotations.append(
                f"No intervalo {df['tempo'].iloc[interval_start]} a {df['tempo'].iloc[len(df) - 1]}, houve rotação")
            tempo_rotacao += df['tempo'].iloc[len(df) - 1] - df['tempo'].iloc[interval_start]
        else:
            rotations.append(
                f"No intervalo {df['tempo'].iloc[interval_start]} a {df['tempo'].iloc[len(df) - 1]}, não houve rotação")

    porcentagem_rotacao = (tempo_rotacao / tempo_total) * 100

    # rotations.append(f"A porcentagem de tempo de rotação é: {porcentagem_rotacao:.2f}%")

    # for rotation in rotations:
    #    print(rotation)

    # if porcentagem_rotacao <= 25:
    #     print("Cavalo calmo.")
    # else:
    #     print("Grande rotação detectada, cavalo inquieto!")

    return porcentagem_rotacao


def calcular_pesos(qtd_movimento, qtd_rotacao, qtd_area, total_area):
    peso_qtd_movimento = 2
    peso_qtd_rotacao = 2
    peso_qtd_area = 2
    peso_total_area = 4

    score = (peso_qtd_movimento * qtd_movimento +
             peso_qtd_rotacao * qtd_rotacao +
             peso_qtd_area * qtd_area +
             peso_total_area * total_area)

    return score

def pessoa(save_folder, output_filename):
    output_path = os.path.join(save_folder, output_filename)

    df = pd.read_excel(output_path)
    
    person_count = df[df['status'] == 'person'].shape[0]
    
    if person_count > 10:
        return True
    else:
        return False


def cavalo(save_folder, output_filename):
    output_path = os.path.join(save_folder, output_filename)
    df = pd.read_excel(output_path)
    
    count_standing_laying = df['status'].isin(['standing', 'laying']).sum()
    
    if count_standing_laying >= 5:
        return True
    else:
        return False
    
def status(save_folder, output_filename):
    output_path = os.path.join(save_folder, output_filename)
    df = pd.read_excel(output_path)
    
    if 'status' not in df.columns:
        raise ValueError("A coluna 'status' não foi encontrada no arquivo.")
    
    standing_count = df['status'].value_counts().get('standing', 0)
    laying_count = df['status'].value_counts().get('laying', 0)
    
    if standing_count > laying_count:
        return True 
    else:
        return False 


def classificar_movimento(save_folder, output_filename, person_output, window_size, threshold):

    hasCavalo = cavalo(save_folder, output_filename)
    hasPessoa = pessoa(save_folder, person_output)
    
    qtd_movimento = None
    total_area = None
    qtd_rotacao = None
    qtd_area = None
    horse_status = None
    score_final = None
    isRestless = None

    if hasCavalo:
        qtd_movimento, total_area = analisar_movimento(save_folder, output_filename)

        qtd_rotacao = detectar_rotacao(save_folder, output_filename, window_size, threshold)

        qtd_area = movimento_area(save_folder, output_filename)

        horse_status = status(save_folder, output_filename)

        print(f"tempo em movimento: {qtd_movimento:.2f}%")
        print(f"tempo em rotacao: {qtd_rotacao:.2f}%")
        print(f"quantificativo de distancia usando área do gráfico: {total_area:.2f} pixels")
        print(f"varicao de area bbox: {qtd_area:.2f}%")

        # Calcula o score final com os pesos
        score_final = calcular_pesos(qtd_movimento, qtd_rotacao, qtd_area, total_area)

        print(f"\nScore final: {score_final:.2f}\n")

        print("----Resultado considerando o Score----\n")

        if score_final >= 300:
            isRestless = True
        else:
            isRestless = False

        if isRestless:
            print("Grande movimentação detectada, cavalo inquieto!")
        else:
            print("Cavalo calmo.")

        print("\n----Resultado sem considerar o Score----\n")

        isRestless = False  # Reinicia o valor para a próxima verificação

        if qtd_movimento >= 30 and total_area >= 100:
            isRestless = True
        elif qtd_movimento >= 30 and qtd_rotacao >= 20:
            isRestless = True
        elif total_area >= 100:
            isRestless = True
        elif qtd_area >= 10:
            isRestless = True
        elif qtd_rotacao >= 20 and qtd_area >= 5:
            isRestless = True

        if isRestless:
            print("\nGrande movimentação detectada, cavalo inquieto!")
        else:
            print("\nCavalo calmo.")
    
    else:
        print("Cavalo não detectado, pulando análises.")

    # Prepara os dados para salvar, mesmo que sejam None
    data = {
        'qtd_movimento (%)': [qtd_movimento],
        'area_movimento': [total_area],
        'qtd_rotacao (%)': [qtd_rotacao],
        'bbox_variation (%)': [qtd_area],
        'score_final': [score_final],
        'inquieto': [score_final is not None and score_final > 300],
        'isStanding': [horse_status], 
        'hasPessoa': [hasPessoa],
        'hasCavalo': [hasCavalo]
    }
    
    df = pd.DataFrame(data)

    # Define o caminho completo do arquivo de saída
    output_path = os.path.join(save_folder, output_filename + "_resultados.xlsx")

    # Salva o DataFrame em um arquivo Excel
    df.to_excel(output_path, index=False)
