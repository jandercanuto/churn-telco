# Telecom Customer Churn Prediction

Modelo preditivo de churn para clientes de telecomunicações com pipeline completo de dados, API de inferência e dashboard interativo. O objetivo é identificar clientes com alta probabilidade de cancelamento antes que o churn ocorra, permitindo ações proativas de retenção.

---

## Problema de Negócio

Em mercados de telecomunicações altamente competitivos, o custo de adquirir um novo cliente é significativamente maior do que o de reter um existente. Este projeto modela a probabilidade de churn de cada cliente e orienta campanhas de retenção direcionadas, maximizando o retorno sobre o investimento da área de CRM.

### Regra de Negócio

| Parâmetro | Valor |
|---|---|
| Custo da campanha (e-mail por cliente) | R$ 15,00 |
| Receita salva por churner retido | R$ 200,00 |
| Público-alvo por ciclo | 1.409 clientes |

O modelo classifica cada cliente como provável churner ou não, e somente os classificados positivamente recebem o e-mail de retenção. O threshold de decisão foi escolhido para maximizar o ROI da campanha, não a acurácia do modelo.

---

## Resultados

| Métrica | Valor |
|---|---|
| Algoritmo | LightGBM (tunado via GridSearchCV) |
| Recall — Cancelou | **0.86** |
| Precision — Cancelou | **0.46** |
| ROI estimado por ciclo | **R$ 54.100** |

> **Por que Recall?** Falsos negativos (churners não identificados) custam receita perdida. Um Recall alto garante que a maior parte dos churners reais seja capturada, mesmo que isso gere alguns falsos positivos — custo aceitável de R$15 por cliente abordado indevidamente.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Armazenamento | PostgreSQL |
| ORM / Conexão | SQLAlchemy |
| Manipulação de dados | pandas, NumPy |
| Feature selection | scorecardpy (IV / WoE) |
| Balanceamento | imbalanced-learn (SMOTE) |
| Modelagem | scikit-learn, LightGBM |
| API de inferência | FastAPI |
| Dashboard | Streamlit |

---

## Arquitetura do Pipeline

```
PostgreSQL
    │
    ▼
Extração (SQLAlchemy)
    │
    ▼
EDA — Análise Exploratória
    │  Distribuições, correlações, missing values
    ▼
Seleção de Variáveis por IV (Information Value)
    │  Critério de Pareto: variáveis com IV ≥ 0.02
    ▼
Encoding (Label / Ordinal)
    │
    ▼
SMOTE — Balanceamento da Classe Minoritária
    │  Aplicado apenas no treino (sem data leakage)
    ▼
LightGBM — Treinamento e Tunagem
    │  GridSearchCV com estratificação
    ▼
FastAPI — Endpoint de Inferência
    │  POST /predict → probabilidade + classe
    ▼
Streamlit — Dashboard Interativo
       Visualização de risco, ROI e distribuição
```

---

## Decisões Técnicas

### Seleção de Variáveis por Information Value
O IV foi usado como critério de Pareto para selecionar apenas as variáveis com poder preditivo real (IV ≥ 0.02). Isso reduz dimensionalidade, evita overfitting em variáveis irrelevantes e melhora a interpretabilidade do modelo.

### SMOTE para Balanceamento
O dataset apresenta desbalanceamento entre classes (maioria `Não Cancelou`). O SMOTE gera amostras sintéticas da classe minoritária **somente no conjunto de treino**, preservando a distribuição real no conjunto de teste e evitando data leakage.

### Exclusão de `TotalCharges`
A variável `TotalCharges` foi excluída por **data leakage**: ela acumula o histórico de pagamentos do cliente e é fortemente correlacionada com o tempo de contrato (`tenure`). Utilizá-la seria equivalente a vazar informação futura para o modelo, inflando artificialmente as métricas.

### Threshold 0.5 Maximiza ROI
Após análise da curva ROI × threshold, o ponto 0.5 foi identificado como ótimo para a regra de negócio vigente (custo R$15 / receita R$200). Outros thresholds foram avaliados e podem ser ajustados caso os parâmetros financeiros mudem.

---

## Estrutura de Pastas

```
churn_telco/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset original (Kaggle)
│   └── processed/
│       ├── telco_churn.csv                         # Dados extraídos do PostgreSQL
│       └── telco_churn_limpo.csv                   # Dados pós-limpeza e encoding
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb               # EDA e seleção de variáveis por IV
│   └── 02_preprocessamento.ipynb                  # Encoding, SMOTE, treino e avaliação
├── scripts/
│   ├── 01_extrair_dados.py                         # Carga no PostgreSQL e extração
│   ├── 03_api.py                                   # FastAPI — endpoint /predict
│   └── 04_frontend.py                             # Dashboard Streamlit
├── models/
│   ├── lgbm_churn.pkl                              # Modelo LightGBM serializado
│   └── scaler.pkl                                  # Scaler serializado
├── requirements.txt
├── requirements-full.txt
└── README.md
```

---

## Como Rodar o Projeto

### 1. Pré-requisitos

- Python 3.10+
- PostgreSQL rodando localmente (ou via Docker)
- `pip` ou `conda`

### 2. Clone e instale as dependências

```bash
git clone https://github.com/seu-usuario/churn_telco.git
cd churn_telco
pip install -r requirements.txt
```

### 3. Configure o banco de dados

Crie um banco PostgreSQL e ajuste a connection string em `scripts/01_extrair_dados.py`:

```python
DATABASE_URL = "postgresql://usuario:senha@localhost:5432/churn_telco"
```

Execute o script de extração para popular o banco e gerar os CSVs processados:

```bash
python scripts/01_extrair_dados.py
```

### 4. Execute os notebooks em ordem

```bash
jupyter notebook notebooks/
```

Rode `01_analise_exploratoria.ipynb` → `02_preprocessamento.ipynb`. O modelo treinado será salvo em `models/lgbm_churn.pkl`.

### 5. Suba a API

```bash
uvicorn scripts.03_api:app --reload
```

A API estará disponível em `http://localhost:8000`. Documentação automática em `http://localhost:8000/docs`.

### 6. Suba o dashboard

```bash
streamlit run scripts/04_frontend.py
```

O dashboard estará disponível em `http://localhost:8501`.

---

## Dataset

[Telco Customer Churn — IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle). Dataset público com 7.043 clientes e 21 variáveis sobre serviços contratados, dados demográficos e comportamento de pagamento.

---

## Autor

**Jander Canuto**
[LinkedIn](https://linkedin.com/in/jandercanuto) · [GitHub](https://github.com/jandercanuto) · jandercanuto@gmail.com
