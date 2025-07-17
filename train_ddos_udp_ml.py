import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Carrega dataset
df = pd.read_csv("./revised_kddcup_dataset.csv", index_col=0)

def train_udp(df, classifier=0):
    """
    Treina modelo para tráfego UDP e exibe métricas de avaliação.
    """
    udp_df = df[df["protocol_type"] == "udp"]

    # Limpa espaços e pontos finais dos rótulos e coloca em minúsculas
    udp_df["result"] = udp_df["result"].str.strip().str.lower().str.replace(".", "", regex=False)

    # Mostra a nova distribuição
    print("Distribuição limpa dos rótulos:")
    print(udp_df["result"].value_counts())

    # Binariza: normal = 1 (benigno), outros = 0 (ataque)
    udp_df["result"] = udp_df["result"].apply(lambda x: 1 if x == "normal" else 0)

    # Codifica o campo 'service'
    service_values = np.unique(udp_df["service"])
    mid = (len(service_values) + 1) / 2
    for i in range(len(service_values)):
        udp_df = udp_df.replace(service_values[i], (i - mid) / 10)

    # Simula a taxa de pacotes por segundo
    # Suponha que cada pacote representa uma amostra a cada 0.1s => 10 pacotes/seg
    udp_df["taxa"] = udp_df["count"] / 10

    udp_features = ["dst_bytes", "service", "src_bytes", "dst_host_srv_count", "taxa"]
    udp_target = "result"
    
    X = udp_df[udp_features]
    y = udp_df[udp_target]

    print("✅ Pré-processamento concluído.")

    # Divide dados para avaliação
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Seleciona o classificador
    if str(classifier) == "0":
        model = KNeighborsClassifier(n_neighbors=3)
        print("🔧 Modelo: KNN (k=3)")
    elif str(classifier) == "1":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        print("🔧 Modelo: Decision Tree")
    else:
        print("⚠️ Modelo inválido. Usando KNN por padrão.")
        model = KNeighborsClassifier(n_neighbors=3)

    # Treina modelo
    model.fit(X_train, y_train)
    print("✅ Modelo treinado com sucesso.")

    # Avaliação
    y_pred = model.predict(X_test)
    print("\n--- 📊 MÉTRICAS DE AVALIAÇÃO ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    # Salva modelo se desejado
    print("💾 Deseja salvar o modelo? (y/n):")
    choice = input().strip().lower()
    if choice == "y":
        pickle.dump(model, open("./udp-server/udp_data.sav", 'wb'))
        print("✅ Modelo salvo em './udp-server/udp_data.sav'.")

# Executa
train_udp(df)
