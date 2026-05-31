#%%
"""
Transformação e validação dos JSONs brutos extraídos da API da Câmara.

Pipeline por tabela:
    1. Carrega o JSON mais recente de data/raw/
    2. Normaliza com pd.json_normalize()
    3. Renomeia colunas para snake_case sem acentos
    4. Seleciona apenas os campos do modelo de dados
    5. Valida: campos obrigatórios, tipos de dado, duplicatas
    6. Retorna DataFrame limpo, pronto para carga no Supabase

Como rodar:
    python src/transform.py                      # transforma tudo
    python src/transform.py --tabela partidos    # só uma tabela
    python src/transform.py --tabela deputados
    python src/transform.py --tabela proposicoes
    python src/transform.py --tabela votacoes
    python src/transform.py --tabela votos
"""

#%%
#Imports
import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd

#Setup
logging.basicConfig(
    level= logging.INFO,
    format= "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RAW_DIR = SRC_DIR.parent / "data" / "raw"
PROCESSED_DIR = SRC_DIR.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
#%%
#UTILITÁRIOS

def remover_acentos(texto: str) -> str:
    """Conver 'Apresentação' -> 'Apresentação'. """
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalizar_nome_coluna(nome: str) -> str:
    """
    Converte qualquer nome de coluna para snake_case sem acentos.

    Casos tratados:
        siglaPartido       → sigla_partido   (camelCase)
        dataApresentação   → data_apresentacao (acento + camelCase)
        proposicao_.id     → proposicao_id   (dot notation do json_normalize)
        deputado_.id       → deputado_id
    """
    nome = remover_acentos(nome)                #1. remove acentos
    nome = nome.replace(".","_")                #2. dots -> _(json_normalize)
    nome = re.sub(r"([A-Z])", r"_\1", nome)     #3. insere _antes de maiúsculas
    nome = nome.lower()                         #4. lowercase
    nome = re.sub(r"_{2,}", "_", nome)          #5. colapsa __ -> _
    nome = nome.strip("_")                      #6. remove _ inicial/final
    return nome 

def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica normalizar_nome_coluna() en tidas as colunas do Data Frame"""
    return df.rename(columns={c: normalizar_nome_coluna(c) for c in df.columns})


def carregar_raw_mais_recente(endpoint: str) -> list[dict]:
    """
    Carrega o JSON mais recente de data/raw/ para o endpoint informado.

    Os arquivos são nomeados {endpoint}_{YYYY-MM-DD}.json — ordenar
    alfabeticamente em ordem decrescente dá o mais recente.

    Levanta
    -------
    FileNotFoundError se nenhum arquivo for encontrado.
    """
    arquivos = sorted(RAW_DIR.glob(f"{endpoint}_????-??-??.json"), reverse=True)

    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo raw para '{endpoint}' em {RAW_DIR}."
            f"Rode extract_raw.py primeiro"
        )
    
    caminho = arquivos[0]
    log.info("Carregando: %s", caminho.name)

    with open(caminho, encoding="utf-8") as f:
        return json.load(f)
    

def selecionar_colunas(df: pd.DataFrame, mapa: dict[str: str], tabela: str) -> pd.DataFrame:
    """
    Seleciona e renomeia colunas conforme o mapa {col_atual: col_final}.

    Se uma coluna do mapa não existir no DataFrame:
        - Loga um aviso (a API pode não retornar todos os campos sempre)
        - Preenche a coluna com None no resultado

    Isso garante que o DataFrame de saída SEMPRE tem exatamente as colunas
    do modelo, mesmo que a API mude ou o campo venha vazio.
    """  

    resultado = {}
    for col_origem, col_destino in mapa.items():
        if col_origem in df.columns:
            resultado[col_destino] = df[col_origem].values
        else:
            log.warning(
                "[%s] Coluna '%s' não encontrada na raw - preenchida com None",
                tabela, col_origem
            )
            resultado[col_destino] = None 
    
    return pd.DataFrame(resultado, index= df.index)


def salvar_processado(tabela: str, df: pd.DataFrame) -> Path:
    """
    Salva o DataFrame transformado em data/processed/ como CSV.
    Nome: {tabela}_{YYYY-MM-DD}.csv
    """
    from datetime import datetime
    hoje    = datetime.today().strftime("%Y-%m-%d")
    caminho = PROCESSED_DIR / f"{tabela}_{hoje}.csv"
    df.to_csv(caminho, index=False, encoding="utf-8")
    log.info("Processado salvo: %s (%d linhas)", caminho.name, len(df))
    return caminho

#%%
#VALIDAÇÕES

def validar_obrigatorios(
    df: pd.DataFrame,
    colunas: list[str],
    tabela: str,
) -> pd.DataFrame:
    """
    Remove registros com valores nulos em colunas obrigatórias.

    Por que remover em vez de falhar?
        Em produção, um registro inválido não deve parar o pipeline inteiro.
        Removemos, logamos, e seguimos — o analista investiga depois pelo log.

    Parâmetros
    ----------
    df      : DataFrame a validar
    colunas : colunas que não podem ter nulo (ex: ["id"])
    tabela  : nome da tabela (para o log)
    """
    n_antes = len(df)

    for col in colunas:
        if col not in df.columns:
            log.warning("[%s] Coluna obrigatória '%s' não existe no DataFrame", tabela, col)
            continue

        nulos = df[col].isna()
        if nulos.any():
            log.warning(
                "[%s] %d registro(s) com '%s' nulo — removidos",
                tabela, nulos.sum(), col
            )
            df = df[~nulos]

    n_removidos = n_antes - len(df)
    if n_removidos:
        log.warning("[%s] Total removidos por nulos obrigatórios: %d", tabela, n_removidos)

    return df

def validar_tipos(
    df: pd.DataFrame,
    specs: dict[str, str],
    tabela: str,
) -> pd.DataFrame:
    """
    Coerce os tipos das colunas conforme as specs informadas.

    Specs suportadas:
        "int"           → pd.Int64 (nullable — aceita None sem virar float)
        "float"         → float64
        "float_positivo"→ float64, zera negativos e loga aviso
        "datetime"      → datetime64[ns], erros viram NaT (logados)
        "str"           → string

    Por que pd.Int64 e não int?
        O tipo nativo int do Python não aceita None. pd.Int64 é o tipo
        inteiro nullable do Pandas — aceita None sem converter para float.

    Parâmetros
    ----------
    df    : DataFrame
    specs : dict {nome_coluna: tipo_esperado}
    tabela: nome da tabela (para o log)
    """
    for col, tipo in specs.items():
        if col not in df.columns:
            log.warning("[%s] Coluna '%s' não encontrada para validar tipo", tabela, col)
            continue

        try:
            if tipo == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            elif tipo == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

            elif tipo == "float_positivo":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                negativos = df[col] < 0
                if negativos.any():
                    log.warning(
                        "[%s] %d valor(es) negativo(s) em '%s' — convertidos para 0",
                        tabela, negativos.sum(), col
                    )
                    df.loc[negativos, col] = 0.0

            elif tipo == "datetime":
                antes = df[col].isna().sum()
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
                depois = df[col].isna().sum()
                if depois > antes:
                    log.warning(
                        "[%s] %d data(s) inválida(s) em '%s' — convertidas para NaT",
                        tabela, depois - antes, col
                    )

            elif tipo == "str":
                df[col] = df[col].astype("string")

        except Exception as exc:
            log.error("[%s] Erro ao converter '%s' para %s: %s", tabela, col, tipo, exc)

    return df


def deduplicar(
    df: pd.DataFrame,
    coluna_id: str,
    tabela: str,
) -> pd.DataFrame:
    """
    Remove registros duplicados pela coluna de id.

    Mantém o primeiro registro encontrado (o mais recente, já que a extração
    ordena por data DESC). Isso garante idempotência — rodar o pipeline duas
    vezes não duplica dados no banco.
    """
    n_antes = len(df)
    df = df.drop_duplicates(subset=[coluna_id], keep="first")
    n_removidos = n_antes - len(df)

    if n_removidos:
        log.warning(
            "[%s] %d duplicata(s) removida(s) pela coluna '%s'",
            tabela, n_removidos, coluna_id
        )
    else:
        log.info("[%s] Sem duplicatas encontradas", tabela)

    return df


def executar_validacoes(
    df: pd.DataFrame,
    tabela: str,
    colunas_obrigatorias: list[str],
    specs_tipos: dict[str, str],
    coluna_dedup: str = "id",
) -> tuple[pd.DataFrame, dict]:
    """
    Orquestra as três validações em sequência e retorna o DataFrame limpo
    junto com um relatório resumido.

    Ordem importa:
        1. Tipos primeiro  → converte antes de checar nulos (evita falsos positivos)
        2. Obrigatórios    → remove registros com campos essenciais nulos
        3. Dedup           → remove duplicatas depois de limpar

    Parâmetros
    ----------
    df                   : DataFrame bruto (após json_normalize e seleção de colunas)
    tabela               : nome da tabela (para o log)
    colunas_obrigatorias : colunas que não podem ser nulas
    specs_tipos          : dict {coluna: tipo_esperado}
    coluna_dedup         : coluna usada para deduplicação (default: "id")

    Retorna
    -------
    (df_limpo, relatorio)
        df_limpo  : DataFrame validado
        relatorio : dict com contagens para auditoria
    """
    n_inicial = len(df)
    log.info("[%s] Iniciando validações | %d registros", tabela, n_inicial)

    df = validar_tipos(df, specs_tipos, tabela)
    df = validar_obrigatorios(df, colunas_obrigatorias, tabela)
    df = deduplicar(df, coluna_dedup, tabela)

    n_final = len(df)
    relatorio = {
        "tabela":        tabela,
        "n_inicial":     n_inicial,
        "n_final":       n_final,
        "n_removidos":   n_inicial - n_final,
        "pct_aprovados": round(n_final / n_inicial * 100, 1) if n_inicial else 0,
    }

    log.info(
        "[%s] Validação concluída | %d → %d registros (%.1f%% aprovados)",
        tabela, n_inicial, n_final, relatorio["pct_aprovados"]
    )

    return df, relatorio

#%%
#TRANSFORMS POR TABELA

def transformar_partidos() -> tuple[pd.DataFrame, dict]:
    """
    Transforma o JSON bruto de /partidos.

    Modelo de destino:
        id         INT  PK
        sigla      TEXT
        nome       TEXT
        url_logo   TEXT  (campo 'uri' da API — link do partido na API,
                          não é o logo. Para o logo real precisaria do
                          endpoint /partidos/{id}. Mapear aqui como placeholder.)
        data_extracao TIMESTAMP

    Nota sobre url_logo:
        O endpoint de listagem retorna apenas 'uri' (link da API).
        O logo real está em /partidos/{id} → campo 'urlLogo'.
        Por ora mapeamos uri → url_logo e enriquecemos depois se necessário.
    """
    TABELA = "partidos"

    MAPA = {
        "id":            "id",
        "sigla":         "sigla",
        "nome":          "nome",
        "uri":           "url_logo",       # placeholder — ver nota acima
        "data_extracao": "data_extracao",
    }

    TIPOS = {
        "id":   "int",
        "sigla": "str",
        "nome":  "str",
    }

    raw  = carregar_raw_mais_recente(TABELA)
    df   = pd.json_normalize(raw)
    df   = normalizar_colunas(df)
    df   = selecionar_colunas(df, MAPA, TABELA)

    return executar_validacoes(
        df,
        tabela=TABELA,
        colunas_obrigatorias=["id", "sigla"],
        specs_tipos=TIPOS,
        coluna_dedup="id",
    )


def transformar_deputados() -> tuple[pd.DataFrame, dict]:
    """
    Transforma o JSON bruto de /deputados.

    Modelo de destino:
        id            INT  PK
        nome          TEXT
        sigla_partido TEXT  FK → partidos.sigla
        uf            TEXT
        email         TEXT
        url_foto      TEXT
        data_extracao TIMESTAMP

    Nota sobre sigla_partido vs id_partido:
        A API retorna a sigla do partido, não o id numérico.
        O relacionamento com a tabela partidos será por sigla.
        Isso é uma decisão de modelagem — sigla é estável o suficiente.
    """
    TABELA = "deputados"

    # Após normalizar_colunas():
    #   siglaPartido → sigla_partido
    #   siglaUf      → sigla_uf      (renomeamos para 'uf' no modelo)
    #   urlFoto      → url_foto
    MAPA = {
        "id":            "id",
        "nome":          "nome",
        "sigla_partido": "sigla_partido",
        "sigla_uf":      "uf",            # siglaUf → sigla_uf → uf (modelo)
        "email":         "email",
        "url_foto":      "url_foto",
        "data_extracao": "data_extracao",
    }

    TIPOS = {
        "id":            "int",
        "nome":          "str",
        "sigla_partido": "str",
        "uf":            "str",
        "email":         "str",
    }

    raw = carregar_raw_mais_recente(TABELA)
    df  = pd.json_normalize(raw)
    df  = normalizar_colunas(df)
    df  = selecionar_colunas(df, MAPA, TABELA)

    return executar_validacoes(
        df,
        tabela=TABELA,
        colunas_obrigatorias=["id", "nome"],
        specs_tipos=TIPOS,
        coluna_dedup="id",
    )


def transformar_proposicoes() -> tuple[pd.DataFrame, dict]:
    """
    Transforma o JSON bruto de /proposicoes.

    Modelo de destino:
        id                INT  PK
        tipo              TEXT   (siglaTipo: PL, PEC, MPV, REQ, etc.)
        numero            INT
        ano               INT
        ementa            TEXT   ← campo que alimenta os embeddings de IA
        data_apresentacao DATE
        tema              TEXT   NULL — preenchido pela camada de IA (Etapa 4)
        resumo_executivo  TEXT   NULL — preenchido pela camada de IA (Etapa 4)
        data_extracao     TIMESTAMP

    Decisão sobre tema e resumo_executivo:
        Criamos as colunas aqui com None. A Etapa 4 vai atualizar essas
        linhas no banco com UPDATE após rodar os embeddings e o LLM.
        Isso mantém o pipeline limpo: Transform não sabe nada de IA.
    """
    TABELA = "proposicoes"

    # Após normalizar_colunas():
    #   siglaTipo        → sigla_tipo (renomeamos para 'tipo' no modelo)
    #   dataApresentacao → data_apresentacao
    MAPA = {
        "id":                  "id",
        "sigla_tipo":          "tipo",              # siglaTipo → sigla_tipo → tipo
        "numero":              "numero",
        "ano":                 "ano",
        "ementa":              "ementa",
        "data_apresentacao":   "data_apresentacao",
        "data_extracao":       "data_extracao",
    }

    TIPOS = {
        "id":                 "int",
        "numero":             "int",
        "ano":                "int",
        "data_apresentacao":  "datetime",
        "ementa":             "str",
    }

    raw = carregar_raw_mais_recente(TABELA)
    df  = pd.json_normalize(raw)
    df  = normalizar_colunas(df)
    df  = selecionar_colunas(df, MAPA, TABELA)

    # Colunas da IA — existem no modelo mas ainda não têm valor
    # A Etapa 4 vai fazer UPDATE nessas colunas
    df["tema"]             = None
    df["resumo_executivo"] = None

    return executar_validacoes(
        df,
        tabela=TABELA,
        colunas_obrigatorias=["id", "ementa"],
        specs_tipos=TIPOS,
        coluna_dedup="id",
    )


def transformar_votacoes() -> tuple[pd.DataFrame, dict]:
    """
    Transforma o JSON bruto de /votacoes.

    Modelo de destino:
        id            TEXT  PK   (a API retorna strings, não ints)
        id_proposicao INT   FK → proposicoes.id
        data          DATE
        descricao     TEXT
        aprovada      BOOL
        data_extracao TIMESTAMP

    Atenção ao campo id_proposicao:
        A API retorna um objeto aninhado 'proposicao_' com o id dentro.
        Após json_normalize + normalizar_colunas():
            proposicao_.id → proposicao_id
        Mapeamos proposicao_id → id_proposicao para seguir a convenção FK.

    Atenção ao campo aprovada:
        A API retorna 'aprovacao' como int (1/0). Convertemos para bool.
    """
    TABELA = "votacoes"

    # Após normalizar_colunas():
    #   proposicao_.id → proposicao_id (renomeamos para id_proposicao — convenção FK)
    #   aprovacao      → aprovacao     (renomeamos para aprovada)
    MAPA = {
        "id":            "id",
        "proposicao_id": "id_proposicao",   # campo aninhado achatado
        "data":          "data",
        "descricao":     "descricao",
        "aprovacao":     "aprovada",
        "data_extracao": "data_extracao",
    }

    TIPOS = {
        "id_proposicao": "int",
        "data":          "datetime",
        "descricao":     "str",
    }

    raw = carregar_raw_mais_recente(TABELA)
    df  = pd.json_normalize(raw)
    df  = normalizar_colunas(df)
    df  = selecionar_colunas(df, MAPA, TABELA)

    # Converte aprovada: 1/0 → True/False
    if "aprovada" in df.columns:
        df["aprovada"] = df["aprovada"].map({1: True, 0: False, "1": True, "0": False})

    return executar_validacoes(
        df,
        tabela=TABELA,
        colunas_obrigatorias=["id"],
        specs_tipos=TIPOS,
        coluna_dedup="id",
    )



def transformar_votos() -> tuple[pd.DataFrame, dict]:
    """
    Transforma o JSON bruto de /votacoes/{id}/votos.

    Modelo de destino:
        id_votacao    TEXT  FK → votacoes.id
        id_deputado   INT   FK → deputados.id
        voto          TEXT  ('Sim', 'Não', 'Abstenção', 'Obstrução', etc.)
        data_extracao TIMESTAMP

    Nota: não há PK própria nessa tabela — a combinação (id_votacao, id_deputado)
    é única. A deduplicação usa os dois campos juntos.

    Atenção ao campo id_deputado:
        A API retorna um objeto aninhado 'deputado_' com o id dentro.
        Após json_normalize + normalizar_colunas():
            deputado_.id → deputado_id
        Mapeamos para id_deputado — convenção FK.
    """
    TABELA = "votos"

    # Após normalizar_colunas():
    #   deputado_.id   → deputado_id   → id_deputado (convenção FK)
    #   tipoVoto       → tipo_voto     → voto (nome mais simples no modelo)
    MAPA = {
        "id_votacao":   "id_votacao",
        "deputado_id":  "id_deputado",   # campo aninhado achatado
        "tipo_voto":    "voto",
        "data_extracao": "data_extracao",
    }

    TIPOS = {
        "id_deputado": "int",
        "voto":        "str",
    }

    raw = carregar_raw_mais_recente(TABELA)
    df  = pd.json_normalize(raw)
    df  = normalizar_colunas(df)
    df  = selecionar_colunas(df, MAPA, TABELA)

    # Para votos, a chave composta (id_votacao + id_deputado) é o identificador único
    # Criamos uma coluna auxiliar para a deduplicação
    df["_chave_dedup"] = df["id_votacao"].astype(str) + "_" + df["id_deputado"].astype(str)

    df, relatorio = executar_validacoes(
        df,
        tabela=TABELA,
        colunas_obrigatorias=["id_votacao", "id_deputado", "voto"],
        specs_tipos=TIPOS,
        coluna_dedup="_chave_dedup",     # dedup pela chave composta
    )

    df = df.drop(columns=["_chave_dedup"])  # remove auxiliar antes de retornar
    return df, relatorio


#%%
#PIPELINE

# Mapeamento nome → função de transform
TRANSFORMS = {
    "partidos":    transformar_partidos,
    "deputados":   transformar_deputados,
    "proposicoes": transformar_proposicoes,
    "votacoes":    transformar_votacoes,
    "votos":       transformar_votos,
}

def rodar_pipeline(tabelas: list[str]) -> dict[str, pd.DataFrame]:
    """
    Executa o transform para cada tabela na lista e retorna um dict
    {nome_tabela: DataFrame limpo}.

    A ordem importa para respeitar as dependências de FK:
        partidos → deputados → proposicoes → votacoes → votos
    """
    resultados    = {}
    relatorios    = []
    ordem_padrao  = ["partidos", "deputados", "proposicoes", "votacoes", "votos"]
    ordem_execucao = [t for t in ordem_padrao if t in tabelas]

    for tabela in ordem_execucao:
        log.info("")
        log.info("━" * 50)
        log.info("TRANSFORMANDO: %s", tabela.upper())
        log.info("━" * 50)

        try:
            df, relatorio = TRANSFORMS[tabela]()
            resultados[tabela] = df
            salvar_processado(tabela, df)
            relatorios.append(relatorio)

        except FileNotFoundError as exc:
            log.error("Arquivo não encontrado para '%s': %s", tabela, exc)
        except Exception as exc:
            log.error("Erro inesperado ao transformar '%s': %s", tabela, exc)

    # Relatório consolidado
    if relatorios:
        log.info("")
        log.info("━" * 50)
        log.info("RELATÓRIO DE VALIDAÇÃO")
        log.info("━" * 50)
        for r in relatorios:
            log.info(
                "  %-15s | inicial: %4d | final: %4d | removidos: %3d | aprovados: %.1f%%",
                r["tabela"], r["n_inicial"], r["n_final"],
                r["n_removidos"], r["pct_aprovados"]
            )

    return resultados

def main():
    parser = argparse.ArgumentParser(
        description="Transforma os JSONs brutos da Câmara em DataFrames validados."
    )
    parser.add_argument(
        "--tabela",
        choices=list(TRANSFORMS.keys()) + ["tudo"],
        default="tudo",
        help="Qual tabela transformar (padrão: tudo)",
    )
    args = parser.parse_args()

    tabelas = list(TRANSFORMS.keys()) if args.tabela == "tudo" else [args.tabela]
    resultados = rodar_pipeline(tabelas)

    log.info("")
    log.info("Transform concluído. DataFrames prontos para carga.")
    log.info("Tabelas processadas: %s", list(resultados.keys()))

    return resultados


if __name__ == "__main__":
    main()