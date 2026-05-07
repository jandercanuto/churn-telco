import streamlit as st  # Importa o Streamlit para construir a interface web
import requests  # Importa requests para fazer chamadas HTTP à API

# Define o título principal exibido no topo da página
st.title("Previsão de Churn - Telecom")

# Exibe uma descrição explicando o propósito do aplicativo
st.markdown(
    "Preencha os dados do cliente abaixo e clique em **Analisar risco de churn** "
    "para saber a probabilidade de cancelamento e receber uma recomendação."
)

st.divider()  # Linha horizontal para separar o cabeçalho do formulário

# Abre o formulário; o botão de submit só dispara o envio ao ser clicado
with st.form("churn_form"):

    st.subheader("Dados contratuais")  # Subtítulo da seção de contrato

    # Selectbox para o tipo de contrato do cliente
    contract = st.selectbox(
        "Tipo de contrato",
        options=["Month-to-month", "One year", "Two year"],
    )

    # Slider para o tempo de permanência do cliente (em meses)
    tenure = st.slider("Tempo como cliente (meses)", min_value=0, max_value=72, value=12)

    # Slider de ponto flutuante para o valor da mensalidade
    monthlycharges = st.slider(
        "Cobranças mensais (R$)", min_value=18.0, max_value=120.0, value=50.0, step=0.5
    )

    st.subheader("Perfil do cliente")  # Subtítulo da seção de perfil

    # Selectbox para indicar se o cliente é idoso; mapeia rótulo para valor numérico
    seniorcitizen_label = st.selectbox("Idoso?", options=["Não", "Sim"])
    seniorcitizen = 0 if seniorcitizen_label == "Não" else 1  # Converte rótulo em 0 ou 1

    # Selectbox para indicar se o cliente tem cônjuge/parceiro
    partner = st.selectbox("Tem parceiro(a)?", options=["Yes", "No"])

    # Selectbox para indicar se o cliente tem dependentes
    dependents = st.selectbox("Tem dependentes?", options=["Yes", "No"])

    st.subheader("Faturamento e pagamento")  # Subtítulo da seção de faturamento

    # Selectbox para indicar se o cliente usa fatura sem papel
    paperlessbilling = st.selectbox("Fatura sem papel?", options=["Yes", "No"])

    # Selectbox para o método de pagamento utilizado pelo cliente
    paymentmethod = st.selectbox(
        "Método de pagamento",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    st.subheader("Serviços contratados")  # Subtítulo da seção de serviços

    # Selectbox para o tipo de serviço de internet
    internetservice = st.selectbox("Serviço de internet", options=["DSL", "Fiber optic", "No"])

    # Selectbox para indicar se o cliente tem segurança online
    onlinesecurity = st.selectbox(
        "Segurança online", options=["No", "Yes", "No internet service"]
    )

    # Selectbox para indicar se o cliente tem suporte técnico
    techsupport = st.selectbox(
        "Suporte técnico", options=["No", "Yes", "No internet service"]
    )

    # Selectbox para indicar se o cliente tem backup online
    onlinebackup = st.selectbox(
        "Backup online", options=["No", "Yes", "No internet service"]
    )

    # Selectbox para indicar se o cliente tem proteção de dispositivo
    deviceprotection = st.selectbox(
        "Proteção de dispositivo", options=["No", "Yes", "No internet service"]
    )

    # Selectbox para indicar se o cliente usa streaming de filmes
    streamingmovies = st.selectbox(
        "Streaming de filmes", options=["No", "Yes", "No internet service"]
    )

    # Selectbox para indicar se o cliente usa streaming de TV
    streamingtv = st.selectbox(
        "Streaming de TV", options=["No", "Yes", "No internet service"]
    )

    # Botão de submit do formulário; retorna True quando clicado
    submitted = st.form_submit_button("Analisar risco de churn")

# Bloco executado somente quando o botão for clicado
if submitted:

    # Monta o dicionário com todos os campos para enviar à API
    payload = {
        "contract": contract,
        "tenure": tenure,
        "monthlycharges": monthlycharges,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "techsupport": techsupport,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "streamingmovies": streamingmovies,
        "streamingtv": streamingtv,
    }

    try:
        # Envia uma requisição POST com os dados em JSON para a API local
        response = requests.post("http://localhost:8000/prever", json=payload, timeout=10)

        # Lança uma exceção se a API retornar um status de erro (4xx ou 5xx)
        response.raise_for_status()

        # Converte a resposta JSON em dicionário Python
        resultado = response.json()

        # Extrai a predição binária (0 = não churn, 1 = churn)
        churn = resultado.get("churn")

        # Extrai a probabilidade de churn retornada pelo modelo
        probabilidade = resultado.get("probabilidade", 0.0)

        # Formata a probabilidade como percentual com uma casa decimal
        prob_pct = f"{probabilidade * 100:.1f}%"

        if churn == 1:
            # Exibe mensagem de alerta em vermelho quando há risco de churn
            st.error(
                f"**Alto risco de churn!** Probabilidade: {prob_pct}\n\n"
                "**Recomendação:** Entre em contato com o cliente, ofereça desconto "
                "ou upgrade de plano para aumentar a fidelização."
            )
        else:
            # Exibe mensagem de sucesso em verde quando o cliente tende a permanecer
            st.success(
                f"**Baixo risco de churn.** Probabilidade: {prob_pct}\n\n"
                "**Recomendação:** Cliente estável. Mantenha a qualidade do serviço "
                "e considere oferecer benefícios de fidelidade."
            )

    except requests.exceptions.ConnectionError:
        # Informa o usuário caso a API não esteja acessível
        st.warning("Não foi possível conectar à API. Verifique se o servidor está rodando em http://localhost:8000.")

    except requests.exceptions.Timeout:
        # Informa o usuário caso a API demore mais de 10 segundos para responder
        st.warning("A API demorou muito para responder. Tente novamente.")

    except requests.exceptions.HTTPError as e:
        # Exibe o erro HTTP retornado pela API (ex: 422 Unprocessable Entity)
        st.error(f"Erro na API: {e}")

    except Exception as e:
        # Captura qualquer outro erro inesperado e exibe a mensagem
        st.error(f"Erro inesperado: {e}")
