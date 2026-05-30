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
    format= "%(asctime)s | %(levename)-8s | %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).resolve().parent()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RAW_DIR = SRC_DIR.parent / "data" / "raw"

#%%
#UTILITÁRIOS

def remover_acentos(texto: str) -> str:
    """Conver 'Apresentação' -> 'Apresentação'. """
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

