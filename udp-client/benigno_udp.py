import socket
import time
import random

server_ip = "server"         # nome do container alvo (resolve na rede Docker)
server_port = 12345          # porta usada pelo servidor UDP

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mensagens = [
    b"Hello",
    b"Ping",
    b"Request",
    b"Check",
    b"KeepAlive"
]

print("Iniciando tráfego UDP benigno...")

for _ in range(20):
    msg = random.choice(mensagens)
    sock.sendto(msg, (server_ip, server_port))
    print(f"Enviado: {msg}")
    time.sleep(random.uniform(0.5, 2))

print("Tráfego benigno finalizado.")
