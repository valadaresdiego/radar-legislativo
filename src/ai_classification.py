#%%
"""
Pipeline:
    1. Carrega modelo de embeddings localmente (sem custo de API)
    2. Gera embeddings para os ~12 temas da consultoria
    3. Carrega proposições sem classificação do Supabase
    4. Para cada proposição, gera embedding da ementa
    5. Calcula similaridade de cosseno com todos os temas
    6. Salva: tema (melhor match), score_tema (confiança) e embedding no banco

Modelo:
    intfloat/multilingual-e5-large  (padrão — ~560MB, excelente para português)
    BAAI/bge-m3                     (alternativa — qualidade superior, mesmo tamanho)

    O modelo é baixado automaticamente do HuggingFace na primeira execução
    e fica em cache local (~/.cache/huggingface). Execuções seguintes são instantâneas.
"""

import argparse
import logging
import os
import sys 
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text 

import torch

#SETUP
logging.basicConfig(
    level = logging.INFO,
    format= "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent 
if str(SRC_DIR) not in sys.path:
    sys.path.inser(0, str(SRC_DIR))

load_dotenv()

#%%
#CONFIGURAÇÃO

MODEL_NAME = "BAAI/bge-m3"
#MODEL_NAME = "intfloat/multilingual-e5-large" segunda opção, tem as mesmas dimensões do bge-m3, não necessita alterar o migration_ai_columns
# Temas da consultoria
# Esses são os rótulos que o modelo vai usar para classificar.

TEMAS = [
    "Tributário e Fiscal",
    "Saúde Pública e Medicamentos",
    "Tecnologia, Inteligência Artificial e Inovação",
    "Meio Ambiente e Sustentabilidade",
    "Trabalho, Emprego e Previdência",
    "Segurança Pública e Sistema Penal",
    "Infraestrutura, Transporte e Saneamento",
    "Educação e Cultura",
    "Economia e Finanças",
    "Direitos Humanos e Cidadania",
    "Agropecuária e Alimentação",
    "Administração Pública e Reforma do Estado",
]
# Tamanho do lote para encoding — ajuste conforme RAM disponível
# CPU: 16-32 é seguro. GPU: pode aumentar para 64-128.
BATCH_SIZE = 64

#%%
#EMBEDDINGS

def carregar_modelo() -> SentenceTransformer:
    """
    Carrega o modelo de embeddings.

    Na primeira execução, baixa o modelo do HuggingFace (~560MB).
    Execuções seguintes carregam do cache local — rápido.

    Por que multilingual-e5-large?
        - Suporte nativo a português
        - 1024 dimensões — representação rica
        - Sem custo de API — roda localmente
        - Boa separação semântica entre temas legislativos
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Carregando modelo: %s | device: %s", MODEL_NAME, device)
    log.info("(Primeira execução faz download de ~560MB do HuggingFace - aguarde)")

    model = SentenceTransformer(MODEL_NAME)

    log.info("Modelo carregado. Dimensão dos embeddings: %d | Device: %s", 
             model.get_sentence_embedding_dimension(), device)
    return model 

def gerar_embeddings_temas(model: SentenceTransformer) -> np.ndarray:
    """
    Gera embeddings para cada tema da lista TEMAS.

    Prefixo "query: " é o formato esperado pelo BAAI/bge-m3 e multilingual-e5 para o
    lado da "consulta" na comparação semântica.

    Retorna
    -------
    np.ndarray shape (n_temas, 1024) — já normalizado (norma L2 = 1)
    """
    log.info("Gerando embeddings para %d temas...", len (TEMAS))

    textos = [f"query: {tema}" for tema in TEMAS]
    embeddings = model.encode(
        textos,
        normalize_embeddings=True, #vetores normalizados -> cosseno = produto escalar
        show_progress_bar=False
    )

    log.info("Embeddings de temas gerados. Shape: %s", embeddings.shape)
    return embeddings

def classificar_lote(ementas: list[str], emb_temas: np.ndarray, model: SentenceTransformer) -> list[tuple[str, float]]:
    """
    Classifica um lote de ementas.

    Para cada ementa:
        1. Gera embedding com prefixo "passage:" (lado do documento)
        2. Calcula similaridade de cosseno com todos os temas
           (= produto escalar, pois ambos estão normalizados)
        3. Retorna o tema com maior score e o score de confiança

    Parâmetros
    ----------
    ementas   : lista de textos de ementa
    emb_temas : embeddings dos temas (shape: n_temas * 1024)
    model     : modelo carregado

    Retorna
    -------
    list de (tema: str, score: float) — um por ementa
    """
    textos = [f"passage: {e}" for e in ementas]

    emb_ementas = model.encode(
        textos,
        normalize_embeddings=True,
        batch_size= BATCH_SIZE,
        show_progress_bar=False
    )

    #Produto escalar de vetores normalizados = similaridade de cosseno
    #Shape resultado: (n_ementas, n_temas)
    
    score_matrix = emb_ementas @ emb_temas.T

    resultados = []
    for scores in score_matrix:
        idx_melhor = int(np.argmax(scores))
        resultados.append((
            TEMAS[idx_melhor],
            float(scores[idx_melhor])
        ))
    return resultados


def embedding_para_pgvector(embedding: np.ndarray) -> str:
    """
    Converte numpy array para o formato string do pgvector.

    pgvector espera: '[0.123, -0.456, 0.789, ...]'
    Não requer a biblioteca pgvector instalada no Python —
    o cast ::vector é feito diretamente no SQL.
    """
    return "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"

#%%
#BANCO DE DADOS

def criar_engine():
    """Cria engine SQLAlchemy com conexão ao Supabase."""
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        raise ValueError("SUPABASE_DB_URL não encontrada no .env")

    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    log.info("✅ Conexão com Supabase estabelecida")
    return engine


def carregar_proposicoes(engine, reclassificar: bool, limite: int | None) -> list[dict]:
    """
    Carrega proposições do banco que ainda precisam de classificação.

    Parâmetros
    ----------
    reclassificar : se True, recarrega todas (inclusive já classificadas)
    limite        : cap de registros (útil para testes)
    """
    where = "" if reclassificar else "WHERE tema IS NULL"
    limit = f"LIMIT {limite}" if limite else ""

    query = f"""
        SELECT id, ementa
        FROM proposicoes
        {where}
        AND ementa IS NOT NULL
        AND ementa != ''
        ORDER BY id
        {limit}
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    proposicoes = [{"id": r[0], "ementa": r[1]} for r in rows]
    log.info("Proposições a classificar: %d", len(proposicoes))
    return proposicoes


def salvar_classificacoes(
    engine,
    classificacoes: list[dict],
    dry_run: bool,
) -> int:
    """
    Faz UPDATE em lote no Supabase com tema, score e embedding.

    Por que UPDATE e não upsert completo?
        O load.py já inseriu as linhas em proposicoes com tema=NULL.
        Aqui só atualizamos as colunas de IA — não mexemos nos outros campos.
        Isso também significa que rodar o load.py depois NÃO apaga a
        classificação (load.py atualiza todas as colunas, mas UPDATE aqui
        é explícito — há conflito. Veja a nota em load.py sobre --preservar-ia).

    Parâmetros
    ----------
    classificacoes : list de dicts com id, tema, score_tema, embedding_str
    dry_run        : se True, apenas loga sem gravar
    """
    if dry_run:
        log.info("[dry-run] %d classificações seriam salvas", len(classificacoes))
        for c in classificacoes[:5]:
            log.info(
                "  id=%-8s | tema=%-45s | score=%.4f",
                c["id"], c["tema"], c["score_tema"]
            )
        if len(classificacoes) > 5:
            log.info("  ... e mais %d", len(classificacoes) - 5)
        return 0

    # UPDATE em lote com embedding como pgvector
    # O cast ::vector converte a string para o tipo nativo do banco
    sql = text("""
        UPDATE proposicoes
        SET
            tema       = :tema,
            score_tema = :score_tema,
            embedding  = CAST(:embedding AS vector)
        WHERE id = :id
    """)

    total = 0
    LOTE  = 100

    for i in range(0, len(classificacoes), LOTE):
        lote = classificacoes[i : i + LOTE]

        with engine.begin() as conn:
            for c in lote:
                conn.execute(sql, {
                    "id":        c["id"],
                    "tema":      c["tema"],
                    "score_tema": c["score_tema"],
                    "embedding": c["embedding_str"],
                })

        total += len(lote)
        log.info(
            "Salvo: %d/%d proposições",
            min(i + LOTE, len(classificacoes)),
            len(classificacoes),
        )

    return total

#%%PIPELINE PRINCIPAL

def rodar_pipeline(
    reclassificar: bool = False,
    limite: int | None  = None,
    dry_run: bool       = False,
):
    """
    Orquestra o pipeline completo de classificação.

    Fluxo:
        1. Carrega modelo (uma vez)
        2. Gera embeddings dos temas (uma vez, ~12 vetores)
        3. Para cada lote de proposições:
           a. Carrega ementas do banco
           b. Gera embeddings
           c. Classifica por similaridade
           d. Salva no banco
    """
    log.info("=" * 55)
    log.info("CLASSIFICAÇÃO TEMÁTICA POR EMBEDDINGS")
    log.info("Modelo: %s", MODEL_NAME)
    log.info("Temas (%d): %s", len(TEMAS), ", ".join(TEMAS))
    log.info("=" * 55)

    # --- 1. Modelo e temas ---
    model      = carregar_modelo()
    emb_temas  = gerar_embeddings_temas(model)

    # --- 2. Proposições do banco ---
    engine       = criar_engine()
    proposicoes  = carregar_proposicoes(engine, reclassificar, limite)

    if not proposicoes:
        log.info("Nenhuma proposição para classificar. Encerrando.")
        return

    # --- 3. Classificação em lotes ---
    log.info("Iniciando classificação em lotes de %d...", BATCH_SIZE)

    classificacoes = []
    total          = len(proposicoes)

    for i in range(0, total, BATCH_SIZE):
        lote_props = proposicoes[i : i + BATCH_SIZE]
        ementas    = [p["ementa"] for p in lote_props]

        resultados = classificar_lote(ementas, emb_temas, model)

        # Monta os dicts para o banco — gera embedding por proposição
        emb_ementas = model.encode(
            [f"passage: {e}" for e in ementas],
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        for prop, (tema, score), embedding in zip(lote_props, resultados, emb_ementas):
            classificacoes.append({
                "id":           prop["id"],
                "tema":         tema,
                "score_tema":   round(score, 6),
                "embedding_str": embedding_para_pgvector(embedding),
            })

        n_processados = min(i + BATCH_SIZE, total)
        log.info(
            "Classificados: %d/%d | último tema: %s (score=%.3f)",
            n_processados, total, resultados[-1][0], resultados[-1][1]
        )

    # --- 4. Salva no banco ---
    log.info("Salvando classificações no Supabase...")
    n_salvo = salvar_classificacoes(engine, classificacoes, dry_run)

    # --- 5. Resumo ---
    log.info("")
    log.info("=" * 55)
    log.info("RESUMO DA CLASSIFICAÇÃO")
    log.info("=" * 55)

    from collections import Counter
    contagem = Counter(c["tema"] for c in classificacoes)
    for tema, n in sorted(contagem.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        log.info("  %-45s %3d (%.1f%%)", tema, n, pct)

    scores = [c["score_tema"] for c in classificacoes]
    log.info("")
    log.info("Score médio de confiança : %.4f", np.mean(scores))
    log.info("Score mínimo             : %.4f", np.min(scores))
    log.info("Score máximo             : %.4f", np.max(scores))
    log.info("")
    log.info("%s %d proposições classificadas", "Simulado:" if dry_run else "Salvas:", n_salvo or total)

#%%
#ENTRYPOINT


def main():
    parser = argparse.ArgumentParser(
        description="Classifica proposições por tema usando embeddings."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Limita o número de proposições (ex: --limite 10 para teste)",
    )
    parser.add_argument(
        "--reclassificar",
        action="store_true",
        help="Reclassifica proposições já classificadas",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o resultado sem gravar no banco",
    )
    args = parser.parse_args()

    rodar_pipeline(
        reclassificar=args.reclassificar,
        limite=args.limite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
