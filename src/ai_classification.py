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

#SETUP
logging.basicConfig(
    level = loggin.info,
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

MODELO_NAME = "BAAI/bge-m3"
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