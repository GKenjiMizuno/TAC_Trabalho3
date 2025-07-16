from scapy.all import IP, UDP, send
import random
import time

server_ip = "server"         # Nome ou IP do container do servidor
server_port = 12345

mensagens = [
    b"Hello",
    b"Ping",
    b"Request",
    b"Check",
    b"KeepAlive"
]

# Lista de IPs de origem falsos (simulados)
fake_source_ips = [
    "10.0.0.101",
    "10.0.0.102",
    "10.0.0.103",
    "10.0.0.104",
    "10.0.0.105"
]

print("Iniciando tráfego UDP benigno com IPs variados...")

for _ in range(20):
    fake_ip = random.choice(fake_source_ips)
    payload = random.choice(mensagens).decode() + "|" + fake_ip    

    pkt = IP(src=fake_ip, dst=server_ip) / UDP(dport=server_port) / payload
    send(pkt, verbose=False)

    print(f"Enviado de {fake_ip} → {server_ip}: {payload}")
    time.sleep(random.uniform(0.5, 2))

print("Tráfego benigno finalizado.")
