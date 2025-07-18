IA para detecção de ataque DDoS UDP Flood

Como executar:

python3 train_ddos_udp_ml.py <5 argumentos de tipo float>

# Ambiente de Teste DDoS UDP com Docker

Este projeto simula um ambiente de rede com três serviços:

- `server`: servidor que escuta pacotes UDP na porta 12345.
- `attacker`: cliente que simula um ataque UDP flood.
- `benign`: cliente que envia pacotes UDP normais, com intervalos regulares.

## Requisitos

- Docker
- Docker Compose (`docker-compose` ou `docker compose`)

## Estrutura
.
├── docker-compose.yml
├── udp-server/
│ ├── Dockerfile
│ └── server.py
└── udp-client/
├── ataque_udp.py
└── benigno_udp.py


## Como usar

### 1. Clonar o projeto ou navegar até a pasta

```bash
cd TAC_Trabalho3
```

### 2. Construir os serviços
```bash
docker-compose build
```

### 3. Rodar o servidor com o cliente benigno
```bash
docker-compose up server benign
```

### 4. Rodar o servidor com o cliente atacante (UDP flood)
```bash
docker-compose up server attacker
```

### 5. Parar os serviços
```bash
docker-compose down
```
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up --force-recreate