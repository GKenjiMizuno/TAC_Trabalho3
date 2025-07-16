# udp_server.py
import socket
from datetime import datetime
import pickle
import pandas as pd

# Carrega modelo treinado
modelo = pickle.load(open("udp_data.sav", "rb"))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 12345))

print("Servidor escutando na porta 12345...")

# Inicialização de contadores simples (ajuste conforme necessidade)
contador_total = 0
contador_por_ip = {}

while True:
    data, addr = sock.recvfrom(4096)
    agora = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    tam = len(data)

    try:
        msg = data.decode()
        # Detecta o IP de origem real do payload, se você embutir no payload (ex: "Hello|10.0.0.101")
        partes = msg.split("|")
        if len(partes) == 2:
            msg, ip_origem = partes[0], partes[1]
        else:
            ip_origem = addr[0]  # fallback
        contador_total += 1
        contador_por_ip[ip_origem] = contador_por_ip.get(ip_origem, 0) + 1

        # Simula extração de features reais (substitua por análise de payload real)
        amostra = {
            "dst_bytes": tam,
            "service": 0.1,  # você pode usar um map de serviços reais (e.g. DNS, NTP)
            "src_bytes": len(msg.encode()),
            "dst_host_srv_count": contador_por_ip[ip_origem],
            "count": contador_total
        }

        print("→ DEBUG | Entrada modelo:", amostra)

        entrada_modelo = pd.DataFrame([amostra])
        predicao = modelo.predict(entrada_modelo)[0]
        
        label = "🟢 BENIGNO" if predicao == 0 else "🔴 ATAQUE"
        print(f"[{agora}] {label} | {ip_origem}:{addr[1]} → \"{msg}\" ({tam} bytes)")

    except UnicodeDecodeError:
        # Pacote com dados binários – possível ataque
        print(f"[{agora}] 🔴 ATAQUE? | {ip_origem}:{addr[1]} → [BINÁRIO] ({tam} bytes)")
