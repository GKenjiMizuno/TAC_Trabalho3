FROM python:3.9-slim


WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir scapy

CMD ["python3", "benigno_udp.py"]
