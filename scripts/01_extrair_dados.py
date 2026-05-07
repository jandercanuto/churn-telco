from pathlib import Path          # manipulação de caminhos de forma portável entre sistemas operacionais
import pandas as pd               # leitura de dados via SQL e exportação para CSV
from sqlalchemy import create_engine  # criação da conexão com o banco de dados via SQLAlchemy

# --- Parâmetros de conexão ---
HOST = "localhost"       # endereço do servidor PostgreSQL
DATABASE = "churn_telco" # nome do banco de dados
USER = "churn_user"      # usuário com acesso ao banco
PASSWORD = "churn123"    # senha do usuário

# Monta a URL de conexão no formato exigido pelo SQLAlchemy com driver psycopg2
# Formato: postgresql+psycopg2://usuario:senha@host/banco
DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}/{DATABASE}"

# Cria o engine do SQLAlchemy, que gerencia o pool de conexões com o banco
engine = create_engine(DATABASE_URL)

# Define o caminho de saída do CSV relativo à pasta raiz do projeto
# Path(__file__) aponta para este script; .parent sobe para /scripts; outro .parent sobe para a raiz
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "telco_churn.csv"

# Query SQL que extrai todas as linhas e colunas da tabela de dados brutos
QUERY = "SELECT * FROM raw_telco_churn"

print("Conectando ao banco de dados...")

# Abre uma conexão com o banco, executa a query e carrega o resultado em um DataFrame pandas
with engine.connect() as conn:           # abre e fecha a conexão automaticamente ao sair do bloco
    df = pd.read_sql(QUERY, con=conn)    # executa a query e retorna um DataFrame com os resultados

print(f"Dados extraídos: {len(df)} linhas, {len(df.columns)} colunas.")

# Garante que o diretório de destino existe antes de salvar (evita erro de caminho inexistente)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Salva o DataFrame como CSV sem incluir o índice do pandas na primeira coluna
df.to_csv(OUTPUT_PATH, index=False)

print(f"Arquivo salvo em: {OUTPUT_PATH}")
