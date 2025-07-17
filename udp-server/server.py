# udp_server.py
import socket
from datetime import datetime
import pickle
import pandas as pd
import time
from collections import deque

# Carrega modelo treinado
modelo = pickle.load(open("udp_data.sav", "rb"))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 12345))

print("Servidor escutando na porta 12345...")

# Inicialização
contador_por_ip = {}
janela = deque()
tempo_janela = 2  # segundos

while True:
    data, addr = sock.recvfrom(4096)
    agora = time.time()
    horario = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    tam = len(data)

    try:
        msg = data.decode()
        partes = msg.split("|")
        if len(partes) == 2:
            msg, ip_origem = partes[0], partes[1]
        else:
            ip_origem = addr[0]
        contador_por_ip[ip_origem] = contador_por_ip.get(ip_origem, 0) + 1

        # Atualiza janela de tempo
        janela.append(agora)
        while janela and janela[0] < agora - tempo_janela:
            janela.popleft()
        taxa_pacotes = len(janela)

        amostra = {
            "dst_bytes": tam,
            "service": 0.1,
            "src_bytes": len(msg.encode()),
            "dst_host_srv_count": contador_por_ip[ip_origem],
            "taxa": taxa_pacotes  # agora representa a taxa
        }

        entrada_modelo = pd.DataFrame([amostra])
        predicao = modelo.predict(entrada_modelo)[0]

        label = "🟢 BENIGNO" if predicao == 1 else "🔴 ATAQUE"
        print(f"[{horario}] {label} | {ip_origem}:{addr[1]} → \"{msg}\" ({tam} bytes)")

    except UnicodeDecodeError:
        print(f"[{horario}] 🔴 ATAQUE? | {ip_origem}:{addr[1]} → [BINÁRIO] ({tam} bytes)")
