#%%
"""
http_client.py
==============
Utilitários de baixo nível para chamadas à API da Câmara dos Deputados.

Responsabilidades deste módulo:
- Fazer chamadas HTTP com retry e backoff exponencial
- Iterar páginas até esgotar os resultados
- Salvar o JSON bruto em data/raw/ antes de qualquer transformação

NÃO faz:
- Transformação dos dados
- Carga no banco
- Lógica de negócio específica de cada endpoint
"""
# %%
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import requests 

# %%
# ---------------------------------------------------------------------------
# Configuração de logging
# Usamos logging (não print) para que os scripts de produção possam ser
# redirecionados para arquivo ou sistema de log centralizado.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# %%
BASE_URL             = "https://dadosabertos.camara.leg.br/api/v2"
HEADERS              = {"Accept": "application/json"}
ITENS_POR_PAG        = 100 #máximo aceito pela api
SLEEP_ENTRE_CHAMADAS = 0.5 # segundos - respeitando o rate limit da api

# Diretório onde os JSONs serão salvos
# Sobe dois níveis a partir de src/ para chegar na raiz do projeto
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Chamada HTTP com retry e backoff exponencial
def get_com_retry(url: str, params: dict, max_tentativas: int = 3) -> dict:
        """
    Faz GET em `url` com os `params` fornecidos.

    Em caso de falha (timeout, erro de rede, status >= 500), tenta novamente
    com espera exponencial:
        tentativa 1 → erro → espera 2s
        tentativa 2 → erro → espera 4s
        tentativa 3 → desiste e levanta exceção

    Parâmetros
    ----------
    url            : URL completa do endpoint
    params         : dicionário de query string
    max_tentativas : quantas vezes tentar antes de desistir

    Retorna
    -------
    dict com o JSON da resposta (campo raiz — inclui 'dados' e 'links')

    Levanta
    -------
    requests.HTTPError  : se a API retornar status 4xx (erro do cliente)
    RuntimeError        : se esgotar todas as tentativas por erros de rede/5xx
    """
        for tentativa in range(1, max_tentativas + 1):
            try:
                resp = requests.get(url, headers=HEADERS, params=params, timeout=15)

                # Erro do cliente (4xx) — não adianta tentar de novo
                if 400 <= resp.status_code < 500:
                    log.error("Erro do cliente %s em %s | params: %s", resp.status_code, url, params)
                    resp.raise_for_status()

                # Erro do servidor (5xx) — pode ser temporário, vale retry
                if resp.status_code >= 500:
                     raise requests.exceptions.ConnectionError(
                          f"Servidor retornou {resp.status_code}")
                return resp.json()
            
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as exc:
                espera = 2 ** tentativa #2s, 4s, 8s
                if tentativa == max_tentativas:
                     log.error("Esgotadas %d tentativas para %s | Erro: %s", max_tentativas, url, exc)
                     raise RuntimeError(f"Falha após {max_tentativas} tentativas: {exc}") from exc
                
                log.warning(
                     "Tentativa %d/%d falhou (%s). Aguardando %ds...",
                     tentativa, max_tentativas, exc, espera
                )
                time.sleep(espera)

# %%
#Paginação automática
def paginar(path: str, params_extras: dict = {}) -> list[dict]:
    """
    Itera todas as páginas de um endpoint e retorna todos os registros
    concatenados em uma única lista.

    Critério de parada: a página retorna menos itens do que ITENS_POR_PAG,
    o que indica que chegamos na última página.

    Parâmetros
    ----------
    path         : caminho do endpoint (ex: "/partidos", "/deputados")
    params_extras: filtros adicionais (ex: {"dataInicio": "2025-01-01"})

    Retorna
    -------
    list[dict] : todos os registros de todas as páginas
    """

    url = f"{BASE_URL}{path}"
    pagina = 1
    todos = []

    log.info("Iniciando a paginação de %s", path)
    
    while True:
        params = {
             "pagina": pagina,
             "itens": ITENS_POR_PAG,
             **params_extras,
        }

        log.info("Buscando pagina %d...", pagina)
        data = get_com_retry(url, params)
        registros = data.get("dados", [])
        todos.extend(registros)

        log.info("Pagina %d: %d registros (total acumulado: %d)", pagina, len(registros), len(todos))

        #Critério de parada
        if len(registros) < ITENS_POR_PAG:
            log.info("Paginação concluída: %d registros totais em %d página(s)", len(todos), pagina)
            break
        
        pagina += 1
        time.sleep(SLEEP_ENTRE_CHAMADAS)
    return todos
#%%
#Auditoria: adiciona data_extração em cada registro
def adicionar_data_extracao(dados: list[dict]) -> list[dict]:
    """
    Enriquece cada registro com o campo 'data_extracao' (ISO 8601).

    Por que no cliente e não no transform?
        Porque queremos registrar QUANDO o dado foi capturado da API,
        não quando foi processado. Se reprocessarmos o JSON amanhã,
        a data_extracao continua sendo a do momento da extração.

    Exemplo:
        {"id": 1, "sigla": "PT", ...}
        → {"id": 1, "sigla": "PT", ..., "data_extracao": "2025-04-01T06:32:11"}
    """
    
    agora = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return [{**registro, "data_extracao": agora} for registro in dados]


# %%
#Salvamento do JSON bruto
def salvar_raw(endpoint: str, dados: list[dict]) -> Path:
    """
    Salva os dados brutos em data/raw/ como JSON, com timestamp no nome.

    Convenção de nome: {endpoint}_{YYYY-MM-DD}.json
    Ex: partidos_2025-04-01.json

    Se o arquivo do dia já existir (ex: reprocessamento), sobrescreve.

    Parâmetros
    ----------
    endpoint : nome curto do endpoint (ex: "partidos", "deputados")
    dados    : lista de dicionários a salvar

    Retorna
    -------
    Path : caminho completo do arquivo salvo
    """ 

    hoje = datetime.today().strftime("%Y-%m-%d")
    nome = f"{endpoint}_{hoje}.json"
    caminho = RAW_DIR / nome 

    with open(caminho, "w", encoding="utf-8") as f:
         json.dump(dados, f, ensure_ascii=False, indent=2)

    log.info("JSON bruto salvo em: %s (%d registros)", caminho, len(dados))
    return caminho 
# %%
