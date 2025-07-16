import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 12345))

print("Servidor escutando na porta 12345...")

while True:
    data, addr = sock.recvfrom(1024)
    print(f"Recebido {len(data)} bytes de {addr}")
