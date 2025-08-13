import math
import random
import os
import time
import numpy as np
from tabulate import tabulate

# ETAPA 0: DADOS DE REFERÊNCIA (BKS - Best Known Solutions)
BKS = {
    "ta001": 14953, "ta002": 16343, "ta003": 14297, "ta004": 16483, "ta005": 14212,
    "ta006": 14624, "ta007": 14936, "ta008": 15193, "ta009": 15544, "ta010": 14392,
    "ta011": 22358, "ta012": 23881, "ta013": 20873, "ta014": 19916, "ta015": 20196,
    "ta016": 20126, "ta017": 19471, "ta018": 21330, "ta019": 21585, "ta020": 22582,
    "ta021": 34683, "ta022": 32855, "ta023": 34825, "ta024": 33006, "ta025": 35328,
    "ta026": 33720, "ta027": 33992, "ta028": 33388, "ta029": 34798, "ta030": 33174,
    "ta031": 72672, "ta032": 78181, "ta033": 72913, "ta034": 77513, "ta035": 78363,
    "ta036": 75402, "ta037": 73890, "ta038": 73442, "ta039": 70871, "ta040": 78729,
    "ta041": 99674, "ta042": 95669, "ta043": 91791, "ta044": 98454, "ta045": 98164,
    "ta046": 97333, "ta047": 99953, "ta048": 98149, "ta049": 96708, "ta050": 98053,
    "ta051": 136881,"ta052": 129975,"ta053": 127617,"ta054": 131943,"ta055": 130967,
    "ta056": 131760,"ta057": 134222,"ta058": 132990,"ta059": 132599,"ta060": 135985,
    "ta061": 288446,"ta062": 280073,"ta063": 275863,"ta064": 261231,"ta065": 274005,
    "ta066": 267899,"ta067": 275491,"ta068": 270668,"ta069": 284652,"ta070": 282366
}

def carregar_instancia(caminho_arquivo):
    """Lê um arquivo de instância de Flow Shop, lidando com pares de 'máquina tempo'."""
    try:
        with open(caminho_arquivo, 'r') as f:
            linhas = [linha.strip() for linha in f if linha.strip()]
        primeira_linha_dados_idx = 0
        while not linhas[primeira_linha_dados_idx].split()[0].isdigit():
            primeira_linha_dados_idx += 1
        primeira_linha = linhas[primeira_linha_dados_idx].split()
        n_tarefas, m_maquinas = int(primeira_linha[0]), int(primeira_linha[1])
        dados_processamento_str = linhas[primeira_linha_dados_idx+1 : primeira_linha_dados_idx+1+n_tarefas]
        tempos_processamento = []
        for linha in dados_processamento_str:
            if not linha: continue
            partes = linha.split()
            tempos_da_tarefa = [int(partes[k]) for k in range(1, len(partes), 2)]
            tempos_processamento.append(tempos_da_tarefa)
        return n_tarefas, m_maquinas, tempos_processamento
    except (IOError, ValueError, IndexError) as e:
        print(f"Erro ao carregar ou processar o arquivo {caminho_arquivo}: {e}")
        return None, None, None

def calcular_total_flowtime_BLOCKING(sequencia, n_tarefas, m_maquinas, tempos_processamento):
    """Calcula o TEMPO TOTAL DE FLUXO para o problema de BLOCKING Flowshop."""
    if not sequencia or len(sequencia) == 0:
        return float('inf')
    d = [[0] * m_maquinas for _ in range(n_tarefas)]
    id_primeira_tarefa = sequencia[0]
    d[0][0] = tempos_processamento[id_primeira_tarefa][0]
    for j in range(1, m_maquinas):
        d[0][j] = d[0][j-1] + tempos_processamento[id_primeira_tarefa][j]
    for i in range(1, n_tarefas):
        id_tarefa_atual = sequencia[i]
        d[i][0] = d[i-1][1] + tempos_processamento[id_tarefa_atual][0]
        for j in range(1, m_maquinas - 1):
            d[i][j] = max(d[i][j-1], d[i-1][j+1]) + tempos_processamento[id_tarefa_atual][j]
        if m_maquinas > 1:
            j = m_maquinas - 1
            d[i][j] = max(d[i][j-1], d[i-1][j]) + tempos_processamento[id_tarefa_atual][j]
    total_flowtime = sum(d[i][m_maquinas - 1] for i in range(n_tarefas))
    return total_flowtime

def calcular_rpd(resultado_obs, melhor_solucao_literatura):
    """Calcula o Relative Percentage Deviation (RPD)."""
    if melhor_solucao_literatura == 0: return float('inf')
    return ((resultado_obs - melhor_solucao_literatura) / melhor_solucao_literatura) * 100

def heuristica_neh(n_tarefas, m_maquinas, tempos_processamento):
    """Implementa a heurística construtiva de Nawaz, Enscore e Ham (NEH)."""
    soma_tempos = sorted(range(n_tarefas), key=lambda i: sum(tempos_processamento[i]), reverse=True)
    melhor_sequencia = [soma_tempos[0]]
    for i in range(1, n_tarefas):
        tarefa_a_inserir = soma_tempos[i]
        melhor_flowtime_local = float('inf')
        melhor_posicao_local = -1
        for j in range(len(melhor_sequencia) + 1):
            sequencia_teste = melhor_sequencia[:j] + [tarefa_a_inserir] + melhor_sequencia[j:]
            flowtime_teste = calcular_total_flowtime_BLOCKING(sequencia_teste, len(sequencia_teste), m_maquinas, tempos_processamento)
            if flowtime_teste < melhor_flowtime_local:
                melhor_flowtime_local = flowtime_teste
                melhor_posicao_local = j
        melhor_sequencia.insert(melhor_posicao_local, tarefa_a_inserir)
    return melhor_sequencia

def busca_local_interchange(sequencia_inicial, n, m, tempos, tempo_limite):
    """Busca local Interchange com estratégia Best Improvement."""
    start_time = time.time()
    melhor_sequencia = list(sequencia_inicial)
    melhor_custo = calcular_total_flowtime_BLOCKING(melhor_sequencia, n, m, tempos)
    tempo_ate_melhor = 0
    iteracoes = 0
    while time.time() - start_time < tempo_limite:
        iteracoes += 1
        melhor_vizinho_custo = melhor_custo
        melhor_vizinho_seq = melhor_sequencia
        houve_melhora_iteracao = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() - start_time > tempo_limite: break
                vizinho = list(melhor_sequencia)
                vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
                custo_vizinho = calcular_total_flowtime_BLOCKING(vizinho, n, m, tempos)
                if custo_vizinho < melhor_vizinho_custo:
                    melhor_vizinho_custo = custo_vizinho
                    melhor_vizinho_seq = vizinho
                    houve_melhora_iteracao = True
            if time.time() - start_time > tempo_limite: break
        if houve_melhora_iteracao:
            melhor_sequencia = melhor_vizinho_seq
            melhor_custo = melhor_vizinho_custo
            tempo_ate_melhor = time.time() - start_time
        else:
            break # Atingiu o ótimo local
    return melhor_sequencia, melhor_custo, tempo_ate_melhor, iteracoes

def simulated_annealing(sequencia_inicial, n, m, tempos, tempo_limite):
    """Executa o Simulated Annealing a partir de uma solução inicial."""
    start_time = time.time()
    temp_inicial, temp_final, alfa = 1500, 0.1, 0.995

    solucao_atual = list(sequencia_inicial)
    custo_atual = calcular_total_flowtime_BLOCKING(solucao_atual, n, m, tempos)
    
    melhor_solucao = solucao_atual
    melhor_custo = custo_atual
    tempo_ate_melhor = 0
    iteracoes = 0
    
    temp = temp_inicial
    while temp > temp_final and time.time() - start_time < tempo_limite:
        iteracoes += 1
        i, j = random.sample(range(n), 2)
        vizinho = list(solucao_atual)
        vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
        custo_vizinho = calcular_total_flowtime_BLOCKING(vizinho, n, m, tempos)
        delta = custo_vizinho - custo_atual
        if delta < 0 or random.random() < math.exp(-delta / temp):
            solucao_atual, custo_atual = vizinho, custo_vizinho
            if custo_atual < melhor_custo:
                melhor_solucao, melhor_custo = solucao_atual, custo_atual
                tempo_ate_melhor = time.time() - start_time
        temp *= alfa
        
    return melhor_solucao, melhor_custo, tempo_ate_melhor, iteracoes


def processar_grupo(grupo_arquivos, bks, pasta_instancias, tempo_limite):
    """Processa um grupo de instâncias e coleta os resultados para todos os métodos."""
    resultados_interchange, resultados_sa = [], []
    for nome_arquivo in grupo_arquivos:
        print("-" * 50)
        print(f"  Processando {nome_arquivo} (tempo limite: {tempo_limite}s)...")
        caminho = os.path.join(pasta_instancias, nome_arquivo)
        n, m, tempos = carregar_instancia(caminho)
        if n is None: continue
        
        chave_bks = os.path.splitext(nome_arquivo)[0]
        melhor_literatura = bks.get(chave_bks)
        if melhor_literatura is None:
            print(f"    Aviso: BKS não encontrado para {chave_bks}.")
            continue
            
        # Passo 1: Heurística Construtiva NEH
        print("    Executando Heurística NEH...")
        seq_neh = heuristica_neh(n, m, tempos)
        custo_neh = calcular_total_flowtime_BLOCKING(seq_neh, n, m, tempos)
        rpd_neh = calcular_rpd(custo_neh, melhor_literatura)
        print(f"    -> NEH: Custo={custo_neh}, RPD={rpd_neh:.2f}%")

        # Passo 2: Refinamento com Busca Local Interchange
        print("    Executando NEH + Busca Local Interchange...")
        seq_ic, custo_ic, tempo_ic, iter_ic = busca_local_interchange(seq_neh, n, m, tempos, tempo_limite)
        rpd_ic = calcular_rpd(custo_ic, melhor_literatura)
        resultados_interchange.append({"rpd": rpd_ic, "tempo": tempo_ic, "iter": iter_ic})
        ### ALTERAÇÃO AQUI: :.2f para :.4f ###
        print(f"    -> IC: Custo={custo_ic}, RPD={rpd_ic:.2f}%, Tempo={tempo_ic:.4f}s")
        print(f"       Sequência: {seq_ic}")

        # Passo 3: Refinamento com Simulated Annealing
        print("    Executando NEH + Simulated Annealing...")
        seq_sa, custo_sa, tempo_sa, iter_sa = simulated_annealing(seq_neh, n, m, tempos, tempo_limite)
        rpd_sa = calcular_rpd(custo_sa, melhor_literatura)
        resultados_sa.append({"rpd": rpd_sa, "tempo": tempo_sa, "iter": iter_sa})
        ### ALTERAÇÃO AQUI: :.2f para :.4f ###
        print(f"    -> SA: Custo={custo_sa}, RPD={rpd_sa:.2f}%, Tempo={tempo_sa:.4f}s")
        print(f"       Sequência: {seq_sa}")

    return resultados_interchange, resultados_sa

def imprimir_tabelas_de_resultados(tamanho_instancia, resultados, nome_metodo):
    """Imprime as tabelas de RPD e Tempo/Iterações para um método."""
    if not resultados: return
    rpds = [r['rpd'] for r in resultados]
    tempos = [r['tempo'] for r in resultados]
    iteracoes = [r['iter'] for r in resultados]
    tabela_rpd = [[tamanho_instancia, f"{np.min(rpds):.2f}", f"{np.max(rpds):.2f}", f"{np.mean(rpds):.2f}", f"{np.std(rpds):.2f}"]]
    print(f"\n--- Tabela de RPD com {nome_metodo} ---")
    print(tabulate(tabela_rpd, headers=["Tamanho", "RPD Mínimo (%)", "RPD Máximo (%)", "ARPD (%)", "Desvio Padrão RPD (%)"], tablefmt="grid"))
    
    ### ALTERAÇÃO AQUI: :.2f para :.4f no tempo médio ###
    tabela_tempo_iter = [[tamanho_instancia, f"{np.min(tempos):.4f}", f"{np.mean(tempos):.4f}", f"{np.min(iteracoes)}", f"{np.max(iteracoes)}"]]
    print(f"\n--- Tabela de Tempo e Iterações com {nome_metodo} ---")
    print(tabulate(tabela_tempo_iter, headers=["Tamanho", "Tempo Mín. (s)", "Tempo Médio (s)", "Iter. Mínimas", "Iter. Máximas"], tablefmt="grid"))

def main():
    pasta_instancias = 'instancias'
    if not os.path.isdir(pasta_instancias):
        print(f"Erro: A pasta '{pasta_instancias}' não foi encontrada.")
        return
    
    TEMPO_LIMITE_PADRAO = 480 
    
    arquivos = sorted([f for f in os.listdir(pasta_instancias) if f.endswith('.txt')])
    grupos = {
        "20x5":  [f for f in arquivos if "ta00" in f and int(os.path.splitext(f)[0].replace("ta","")) <=10],
        "20x10": [f for f in arquivos if "ta01" in f and int(os.path.splitext(f)[0].replace("ta","")) <=20],
        "20x20": [f for f in arquivos if "ta02" in f and int(os.path.splitext(f)[0].replace("ta","")) <=30],
        "50x5":  [f for f in arquivos if "ta03" in f and int(os.path.splitext(f)[0].replace("ta","")) <=40],
        "50x10": [f for f in arquivos if "ta04" in f and int(os.path.splitext(f)[0].replace("ta","")) <=50],
        "50x20": [f for f in arquivos if "ta05" in f and int(os.path.splitext(f)[0].replace("ta","")) <=60],
        "100x5": [f for f in arquivos if "ta06" in f and int(os.path.splitext(f)[0].replace("ta","")) <=70]
    }
    
    opcoes_menu = {str(i+1): k for i, k in enumerate(grupos.keys())}
    opcoes_menu['8'] = "TODOS os grupos"
    opcoes_menu['9'] = "Sair"
    
    while True:
        print("\n" + "="*20 + " MENU DE EXECUÇÃO " + "="*20)
        for key, val in opcoes_menu.items(): print(f"  {key}. Executar grupo {val}")
        print("-" * 58)
        
        escolha = input("Digite sua escolha de grupo: ")

        if escolha == '9': 
            print("Encerrando o programa.")
            break
        
        grupos_a_processar = []
        if escolha == '8': 
            grupos_a_processar = list(grupos.keys())
        elif escolha in opcoes_menu: 
            grupos_a_processar.append(opcoes_menu[escolha])
        else: 
            print("Opção inválida. Tente novamente.")
            continue
            
        for nome_grupo in grupos_a_processar:
            print(f"\n>>> INICIANDO PROCESSAMENTO DO GRUPO: {nome_grupo} <<<")
            resultados_ic, resultados_sa = processar_grupo(grupos[nome_grupo], BKS, pasta_instancias, TEMPO_LIMITE_PADRAO)
            imprimir_tabelas_de_resultados(nome_grupo, resultados_ic, "NEH + Interchange")
            imprimir_tabelas_de_resultados(nome_grupo, resultados_sa, "NEH + Simulated Annealing")
            print(f">>> FIM DO PROCESSAMENTO DO GRUPO: {nome_grupo} <<<")

if __name__ == "__main__":
    main()