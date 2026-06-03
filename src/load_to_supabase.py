#%%
"""

Carga dos DataFrames transformados no Supabase via SQLAlchemy.

Estratégia de carga:
    UPSERT para todas as tabelas (INSERT ... ON CONFLICT DO UPDATE).
    Isso garante idempotência — rodar duas vezes não duplica registros.

    Dimensões (partidos, deputados):
        Full refresh via upsert — envia todos os registros sempre.
        Volume pequeno (~30 e ~513 registros), sem impacto.

    Fatos (proposicoes, votacoes, votos):
        Upsert incremental — o recorte de data já vem da extração,
        então só chegam registros do período configurado.
        Novos registros são inseridos, existentes são atualizados.

Pré-requisitos:
    1. Tabelas criadas no Supabase via schema.sql
    2. Arquivo .env com SUPABASE_DB_URL configurado
    3. transform.py no mesmo diretório (src/)

Como rodar:
    python src/load_to_supabase.py                    # carrega tudo
    python src/load_to_supabase.py --tabela partidos  # só uma tabela
    python src/load_to_supabase.py --dry-run          # mostra o que seria carregado sem gravar
"""
#%%


import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

#SETUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Carrega variáveis do .env antes de qualquer import que precise delas
load_dotenv()

from transform import (
    transformar_deputados,
    transformar_partidos,
    transformar_proposicoes,
    transformar_votacoes,
    transformar_votos,
)

#%%
#CONEXÃO
def criar_engine():
    """
    Cria o engine SQLAlchemy a partir da variável SUPABASE_DB_URL do .env.
    pool_pre_ping=True:
        Testa a conexão antes de usar — evita erros silenciosos
        quando a conexão do pool expirou.
    """
    url = os.getenv("SUPABASE_DB_URL")

    if not url:
        raise ValueError(
            "SUPABASE_DB_URL não encontrada. "
            "Verifique se o arquivo .env existe e contém a variável."
        )

    engine = create_engine(url, pool_pre_ping=True)

    # Testa a conexão imediatamente — falha rápido com mensagem clara
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("✅ Conexão com Supabase estabelecida")
    except Exception as exc:
        raise ConnectionError(
            f"Não foi possível conectar ao Supabase: {exc}\n"
            f"Verifique a SUPABASE_DB_URL no .env."
        ) from exc

    return engine


# ===========================================================================
# SEÇÃO 2 — UTILITÁRIOS DE CARGA
# ===========================================================================

# Cache de metadados — tabelas refletidas uma vez e reutilizadas
_metadata = MetaData()


def get_tabela(engine, nome: str) -> Table:
    """
    Reflete a tabela do banco e armazena em cache.

    'Refletir' significa ler a estrutura da tabela diretamente do banco
    (colunas, tipos, constraints) em vez de redefinir no Python.
    Isso garante que o Python e o banco estão sempre em sincronia.

    Levanta
    -------
    Exception se a tabela não existir — lembre de rodar schema.sql primeiro.
    """
    if nome not in _metadata.tables:
        try:
            Table(nome, _metadata, autoload_with=engine)
        except Exception:
            raise RuntimeError(
                f"Tabela '{nome}' não encontrada no banco. "
                f"Execute o schema.sql no Supabase SQL Editor primeiro."
            )
    return _metadata.tables[nome]


def df_para_records(df: pd.DataFrame) -> list[dict]:
    """
    Converte DataFrame para lista de dicts compatível com SQLAlchemy.

    Por que astype(object)?
        Pandas tem tipos nullable próprios (Int64, string) que não existem
        em Python nativo. SQLAlchemy não entende esses tipos diretamente.
        Converter para object e substituir NA por None resolve isso.

    Resultado:
        Int64 NA  → None   (não NaN, que é float)
        string NA → None
        float NaN → None
    """
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def upsert_em_lotes(
    engine,
    nome_tabela: str,
    df: pd.DataFrame,
    pk: list[str],
    tamanho_lote: int = 500,
) -> int:
    """
    Faz upsert do DataFrame na tabela em lotes.

    Por que lotes?
        Upserts com milhares de registros em uma única query podem
        estourar o limite de parâmetros do PostgreSQL (~65k) e consumir
        memória excessiva. Lotes de 500 são seguros e eficientes.

    Lógica do upsert:
        INSERT INTO tabela (col1, col2, ...)
        VALUES (...)
        ON CONFLICT (pk) DO UPDATE
            SET col1 = EXCLUDED.col1, col2 = EXCLUDED.col2, ...

        EXCLUDED refere-se aos valores que estavam sendo inseridos
        e causaram o conflito — ou seja, os novos valores.

    Parâmetros
    ----------
    engine        : engine SQLAlchemy
    nome_tabela   : nome da tabela no banco
    df            : DataFrame a carregar
    pk            : lista de colunas que formam a chave primária
    tamanho_lote  : registros por batch (padrão: 500)

    Retorna
    -------
    int : total de linhas afetadas (inseridas + atualizadas)
    """
    if df.empty:
        log.warning("[%s] DataFrame vazio — nada a carregar", nome_tabela)
        return 0

    tabela   = get_tabela(engine, nome_tabela)
    records  = df_para_records(df)
    total    = 0
    n_lotes  = (len(records) + tamanho_lote - 1) // tamanho_lote

    log.info("[%s] Iniciando upsert | %d registros em %d lote(s)", nome_tabela, len(records), n_lotes)

    for i in range(0, len(records), tamanho_lote):
        lote        = records[i : i + tamanho_lote]
        num_lote    = i // tamanho_lote + 1

        try:
            stmt = pg_insert(tabela).values(lote)

            # Colunas que serão atualizadas no conflito (tudo exceto a PK)
            colunas_update = {
                col: stmt.excluded[col]
                for col in df.columns
                if col not in pk
            }

            if colunas_update:
                stmt = stmt.on_conflict_do_update(
                    index_elements=pk,
                    set_=colunas_update,
                )
            else:
                # Tabela só tem PK — apenas ignora conflitos
                stmt = stmt.on_conflict_do_nothing()

            with engine.begin() as conn:
                result = conn.execute(stmt)
                linhas = result.rowcount

            total += linhas
            log.info(
                "[%s] Lote %d/%d | %d linhas afetadas",
                nome_tabela, num_lote, n_lotes, linhas
            )

        except Exception as exc:
            log.error(
                "[%s] Erro no lote %d/%d: %s",
                nome_tabela, num_lote, n_lotes, exc
            )
            raise

    log.info("[%s] ✅ Upsert concluído | %d linhas afetadas no total", nome_tabela, total)
    return total


# ===========================================================================
# SEÇÃO 3 — CARGA POR TABELA
# ===========================================================================

def carregar_partidos(engine, dry_run: bool = False) -> int:
    """
    Estratégia: full refresh via upsert.
    Volume fixo (~30 partidos) — sempre enviamos tudo.
    """
    log.info("━" * 50)
    log.info("CARREGANDO: partidos")
    df, _ = transformar_partidos()

    if dry_run:
        log.info("[dry-run] %d registros seriam carregados", len(df))
        return 0

    return upsert_em_lotes(engine, "partidos", df, pk=["id"])


def carregar_deputados(engine, dry_run: bool = False) -> int:
    """
    Estratégia: full refresh via upsert.
    Volume fixo (~513 deputados) — sempre enviamos tudo.
    """
    log.info("━" * 50)
    log.info("CARREGANDO: deputados")
    df, _ = transformar_deputados()

    if dry_run:
        log.info("[dry-run] %d registros seriam carregados", len(df))
        return 0

    return upsert_em_lotes(engine, "deputados", df, pk=["id"])


def carregar_proposicoes(engine, dry_run: bool = False) -> int:
    """
    Estratégia: upsert incremental.
    Só chegam registros do período extraído (ex: últimos 30 dias).
    Novos → inseridos. Existentes → atualizados.

    Nota sobre tema e resumo_executivo:
        Essas colunas chegam como None aqui. Após a Etapa 4 (IA),
        faremos UPDATE apenas nessas colunas — o upsert aqui não
        vai sobrescrever valores já preenchidos pela IA porque
        a Etapa 4 faz um UPDATE direto no banco, não passa pelo load_to_supabase.py.

        ATENÇÃO: Se você rodar load_to_supabase.py após a Etapa 4, o upsert vai
        sobrescrever tema e resumo_executivo com None novamente.
        Solução: exclua essas colunas do set_ do upsert depois que a
        IA já tiver rodado. Uma flag --preservar-ia pode ser adicionada.
    """
    log.info("━" * 50)
    log.info("CARREGANDO: proposicoes")
    df, _ = transformar_proposicoes()

    if dry_run:
        log.info("[dry-run] %d registros seriam carregados", len(df))
        return 0

    return upsert_em_lotes(engine, "proposicoes", df, pk=["id"])


def carregar_votacoes(engine, dry_run: bool = False) -> int:
    """
    Estratégia: upsert incremental.
    A votações.id é string na API — o PK é texto.
    """
    log.info("━" * 50)
    log.info("CARREGANDO: votacoes")
    df, _ = transformar_votacoes()

    if dry_run:
        log.info("[dry-run] %d registros seriam carregados", len(df))
        return 0

    return upsert_em_lotes(engine, "votacoes", df, pk=["id"])


def carregar_votos(engine, dry_run: bool = False) -> int:
    """
    Estratégia: upsert incremental.
    PK composta: (id_votacao, id_deputado).
    Um deputado tem exatamente um voto por votação.
    """
    log.info("━" * 50)
    log.info("CARREGANDO: votos")
    df, _ = transformar_votos()

    if dry_run:
        log.info("[dry-run] %d registros seriam carregados", len(df))
        return 0

    return upsert_em_lotes(engine, "votos", df, pk=["id_votacao", "id_deputado"])


# ===========================================================================
# SEÇÃO 4 — PIPELINE & ENTRYPOINT
# ===========================================================================

# Mapeamento nome → função de carga
# Ordem respeita dependências lógicas:
#   partidos → deputados → proposicoes → votacoes → votos
CARGAS = {
    "partidos":    carregar_partidos,
    "deputados":   carregar_deputados,
    "proposicoes": carregar_proposicoes,
    "votacoes":    carregar_votacoes,
    "votos":       carregar_votos,
}


def rodar_pipeline(tabelas: list[str], engine, dry_run: bool = False) -> dict:
    """
    Executa a carga para cada tabela na ordem correta.

    Retorna um dict com o total de linhas carregadas por tabela.
    """
    ordem_padrao   = list(CARGAS.keys())
    ordem_execucao = [t for t in ordem_padrao if t in tabelas]
    resultados     = {}

    for tabela in ordem_execucao:
        try:
            n = CARGAS[tabela](engine, dry_run=dry_run)
            resultados[tabela] = n
        except Exception as exc:
            log.error("Falha ao carregar '%s': %s", tabela, exc)
            resultados[tabela] = -1

    # Resumo final
    log.info("")
    log.info("━" * 50)
    log.info("RESUMO DA CARGA%s", " (dry-run)" if dry_run else "")
    log.info("━" * 50)
    for tabela, n in resultados.items():
        status = "dry-run" if dry_run else ("✅" if n >= 0 else "❌ ERRO")
        log.info("  %-15s | %s %s", tabela, n if n >= 0 else "falhou", status)

    return resultados


def main():
    parser = argparse.ArgumentParser(
        description="Carrega os dados transformados no Supabase."
    )
    parser.add_argument(
        "--tabela",
        choices=list(CARGAS.keys()) + ["tudo"],
        default="tudo",
        help="Qual tabela carregar (padrão: tudo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria carregado sem gravar nada no banco",
    )
    args = parser.parse_args()

    tabelas = list(CARGAS.keys()) if args.tabela == "tudo" else [args.tabela]

    if args.dry_run:
        log.info("Modo dry-run ativado — nenhum dado será gravado no banco")
        rodar_pipeline(tabelas, engine=None, dry_run=True)
    else:
        engine = criar_engine()
        rodar_pipeline(tabelas, engine, dry_run=False)


if __name__ == "__main__":
    main()