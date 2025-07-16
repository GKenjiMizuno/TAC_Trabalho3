import socket
import time

server_ip = "server"
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Enviando pacotes UDP benignos...")

mensagens = [
    b"Ping 1",
    b"Ping 2",
    b"Ping 3",
    b"Ultimo pacote"
]

for msg in mensagens:
    sock.sendto(msg, (server_ip, port))
    time.sleep(2)  # Aguarda 2 segundos entre pacotes
