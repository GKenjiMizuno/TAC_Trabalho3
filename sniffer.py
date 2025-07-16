import pyshark

def extrair_features(pacote):
    try:
        udp = pacote.udp
        ip = pacote.ip
        return {
            'src_ip': ip.src,
            'dst_ip': ip.dst,
            'src_port': udp.srcport,
            'dst_port': udp.dstport,
            'length': int(udp.length)
        }
    except AttributeError:
        return None

print("Sniffer ativo - Capturando pacotes UDP...")

capture = pyshark.LiveCapture(interface='eth0', display_filter='udp')

for pkt in capture.sniff_continuously():
    features = extrair_features(pkt)
    if features:
        print("[PACOTE]", features)
        # Chamada para o seu modelo ML
        # resultado = modelo_ml.classificar(features)
        # print(">> Classificação:", resultado)
