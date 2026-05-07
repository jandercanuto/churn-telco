import joblib  # Para carregar os arquivos .pkl do modelo e do scaler
import numpy as np  # Para manipulação de arrays numéricos
import pandas as pd  # Para criar e organizar o DataFrame com os dados do cliente
from fastapi import FastAPI  # Framework principal da API
from pydantic import BaseModel  # Para validar e tipar os dados recebidos no POST

# Inicializa a aplicação FastAPI
app = FastAPI(title="Churn Predictor API", description="Prevê churn de clientes de telecom")

# Carrega o modelo LightGBM treinado a partir do arquivo .pkl
model = joblib.load("/home/jander/churn_telco/models/lgbm_churn.pkl")

# Carrega o MinMaxScaler treinado para normalizar tenure e monthlycharges
scaler = joblib.load("/home/jander/churn_telco/models/scaler.pkl")

# Define o schema de entrada com todos os campos esperados pelo POST /prever
class ClienteInput(BaseModel):
    contract: str           # Tipo de contrato: "Month-to-month", "One year" ou "Two year"
    tenure: float           # Tempo de permanência do cliente em meses
    monthlycharges: float   # Valor cobrado mensalmente
    paperlessbilling: str   # Fatura sem papel: "Yes" ou "No"
    dependents: str         # Possui dependentes: "Yes" ou "No"
    partner: str            # Possui parceiro: "Yes" ou "No"
    seniorcitizen: int      # É idoso: 0 (não) ou 1 (sim)
    onlinesecurity: str     # Segurança online: "Yes", "No" ou "No internet service"
    techsupport: str        # Suporte técnico: "Yes", "No" ou "No internet service"
    internetservice: str    # Tipo de internet: "DSL", "Fiber optic" ou "No"
    onlinebackup: str       # Backup online: "Yes", "No" ou "No internet service"
    deviceprotection: str   # Proteção de dispositivo: "Yes", "No" ou "No internet service"
    paymentmethod: str      # Método de pagamento: "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check" ou "Mailed check"
    streamingmovies: str    # Streaming de filmes: "Yes", "No" ou "No internet service"
    streamingtv: str        # Streaming de TV: "Yes", "No" ou "No internet service"


# Mapeamentos de label encoding para variáveis categóricas ordinais
LABEL_ENCODING = {
    "contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "paperlessbilling": {"No": 0, "Yes": 1},
    "dependents": {"No": 0, "Yes": 1},
    "partner": {"No": 0, "Yes": 1},
}

# Ordem exata das colunas que o modelo espera receber
FEATURE_ORDER = [
    "contract",
    "tenure",
    "monthlycharges",
    "paperlessbilling",
    "dependents",
    "partner",
    "seniorcitizen",
    "onlinesecurity_No internet service",
    "onlinesecurity_Yes",
    "techsupport_No internet service",
    "techsupport_Yes",
    "internetservice_Fiber optic",
    "internetservice_No",
    "onlinebackup_No internet service",
    "onlinebackup_Yes",
    "deviceprotection_No internet service",
    "deviceprotection_Yes",
    "paymentmethod_Credit card (automatic)",
    "paymentmethod_Electronic check",
    "paymentmethod_Mailed check",
    "streamingmovies_No internet service",
    "streamingmovies_Yes",
    "streamingtv_No internet service",
    "streamingtv_Yes",
]


def preprocessar(cliente: ClienteInput) -> pd.DataFrame:
    """Converte os dados brutos do cliente no vetor de features esperado pelo modelo."""

    # Aplica label encoding nas variáveis categóricas ordinais
    contract_enc = LABEL_ENCODING["contract"][cliente.contract]
    paperlessbilling_enc = LABEL_ENCODING["paperlessbilling"][cliente.paperlessbilling]
    dependents_enc = LABEL_ENCODING["dependents"][cliente.dependents]
    partner_enc = LABEL_ENCODING["partner"][cliente.partner]

    # Normaliza tenure e monthlycharges com o scaler previamente treinado
    valores_para_escalar = np.array([[cliente.tenure, cliente.monthlycharges]])  # Shape (1, 2) exigido pelo scaler
    tenure_norm, monthlycharges_norm = scaler.transform(valores_para_escalar)[0]  # Extrai os dois valores normalizados

    # Gera as colunas one-hot para onlinesecurity (referência: "No" com drop_first=True)
    onlinesecurity_no_internet = 1 if cliente.onlinesecurity == "No internet service" else 0
    onlinesecurity_yes = 1 if cliente.onlinesecurity == "Yes" else 0

    # Gera as colunas one-hot para techsupport
    techsupport_no_internet = 1 if cliente.techsupport == "No internet service" else 0
    techsupport_yes = 1 if cliente.techsupport == "Yes" else 0

    # Gera as colunas one-hot para internetservice (referência: "DSL" com drop_first=True)
    internetservice_fiber = 1 if cliente.internetservice == "Fiber optic" else 0
    internetservice_no = 1 if cliente.internetservice == "No" else 0

    # Gera as colunas one-hot para onlinebackup
    onlinebackup_no_internet = 1 if cliente.onlinebackup == "No internet service" else 0
    onlinebackup_yes = 1 if cliente.onlinebackup == "Yes" else 0

    # Gera as colunas one-hot para deviceprotection
    deviceprotection_no_internet = 1 if cliente.deviceprotection == "No internet service" else 0
    deviceprotection_yes = 1 if cliente.deviceprotection == "Yes" else 0

    # Gera as colunas one-hot para paymentmethod (referência: "Bank transfer (automatic)" com drop_first=True)
    paymentmethod_credit = 1 if cliente.paymentmethod == "Credit card (automatic)" else 0
    paymentmethod_echeck = 1 if cliente.paymentmethod == "Electronic check" else 0
    paymentmethod_mailed = 1 if cliente.paymentmethod == "Mailed check" else 0

    # Gera as colunas one-hot para streamingmovies
    streamingmovies_no_internet = 1 if cliente.streamingmovies == "No internet service" else 0
    streamingmovies_yes = 1 if cliente.streamingmovies == "Yes" else 0

    # Gera as colunas one-hot para streamingtv
    streamingtv_no_internet = 1 if cliente.streamingtv == "No internet service" else 0
    streamingtv_yes = 1 if cliente.streamingtv == "Yes" else 0

    # Monta o dicionário com todas as features na ordem correta
    features = {
        "contract": contract_enc,
        "tenure": tenure_norm,
        "monthlycharges": monthlycharges_norm,
        "paperlessbilling": paperlessbilling_enc,
        "dependents": dependents_enc,
        "partner": partner_enc,
        "seniorcitizen": cliente.seniorcitizen,
        "onlinesecurity_No internet service": onlinesecurity_no_internet,
        "onlinesecurity_Yes": onlinesecurity_yes,
        "techsupport_No internet service": techsupport_no_internet,
        "techsupport_Yes": techsupport_yes,
        "internetservice_Fiber optic": internetservice_fiber,
        "internetservice_No": internetservice_no,
        "onlinebackup_No internet service": onlinebackup_no_internet,
        "onlinebackup_Yes": onlinebackup_yes,
        "deviceprotection_No internet service": deviceprotection_no_internet,
        "deviceprotection_Yes": deviceprotection_yes,
        "paymentmethod_Credit card (automatic)": paymentmethod_credit,
        "paymentmethod_Electronic check": paymentmethod_echeck,
        "paymentmethod_Mailed check": paymentmethod_mailed,
        "streamingmovies_No internet service": streamingmovies_no_internet,
        "streamingmovies_Yes": streamingmovies_yes,
        "streamingtv_No internet service": streamingtv_no_internet,
        "streamingtv_Yes": streamingtv_yes,
    }

    # Cria um DataFrame com uma linha e as colunas na ordem exata esperada pelo modelo
    df = pd.DataFrame([features], columns=FEATURE_ORDER)

    return df  # Retorna o DataFrame pronto para predição


@app.post("/prever")  # Rota POST que recebe os dados do cliente e retorna a previsão
def prever(cliente: ClienteInput):
    """Recebe os dados de um cliente e retorna a previsão de churn."""

    # Pré-processa os dados do cliente para o formato esperado pelo modelo
    X = preprocessar(cliente)

    # Calcula a probabilidade de churn (classe 1) usando o modelo LightGBM
    probabilidade = float(model.predict_proba(X)[0][1])  # predict_proba retorna [[prob_0, prob_1]]

    # Determina a classe prevista: 1 (churn) se probabilidade >= 0.5, caso contrário 0
    churn = int(probabilidade >= 0.5)

    # Define a recomendação de ação com base na probabilidade calculada
    if probabilidade >= 0.5:
        recomendacao = "Cliente com alto risco de churn. Enviar oferta de retenção."  # Cliente propenso a cancelar
    else:
        recomendacao = "Cliente com baixo risco de churn. Nenhuma ação necessária."  # Cliente provavelmente vai ficar

    # Retorna o resultado como JSON com churn, probabilidade e recomendação
    return {
        "churn": churn,                                          # 0 = não vai cancelar, 1 = vai cancelar
        "probabilidade": round(probabilidade, 4),                # Probabilidade arredondada a 4 casas decimais
        "recomendacao": recomendacao,                            # Texto com a ação sugerida
    }
