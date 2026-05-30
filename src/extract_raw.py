#%%
"""
extract_raw.py
====================
Extração dos endpoints de dimensão: /partidos e /deputados.

Por que "dimensões"?
    Partidos e deputados são tabelas de referência — dados que mudam pouco
    e que outras tabelas (votações, votos, despesas) referenciam via chave.
    Faz sentido extraí-los juntos e antes dos dados fato.

Como rodar:
    python src/extract_raw.py

    Flags opcionais:
        --endpoint partidos     → extrai só partidos
        --endpoint deputados    → extrai só deputados
        (sem flag)              → extrai tudo

Saída:
    data/raw/partidos_YYYY-MM-DD.json
    data/raw/deputados_YYYY-MM-DD.json
    data/raw/deputados_detalhes_YYYY-MM-DD.json  (campos extras de cada deputado)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Garante que src/ está no sys.path independente de onde o script é chamado
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camara_client import(
    BASE_URL,
    SLEEP_ENTRE_CHAMADAS,
    get_com_retry,
    paginar,
    salvar_raw,
)

log = logging.getLogger(__name__)

#%%
#Helpers de periodo
def calcular_periodo(dias: int) -> tuple[str, str]:
    """
    Retorna (data_inicio, data_fim) no formato YYYY-MM-DD
    para os últimos `dias` dias até hoje.
    """
    data_fim    = datetime.today()
    data_inicio = data_fim - timedelta(days=dias)
    return data_inicio.strftime("%Y-%m-%d"), data_fim.strftime("%Y-%m-%d")


# %%
#Extração `Partidos`

def extrair_partidos() -> list[dict]:
    """
    Extrai todos os partidos registrados na Câmara.

    O endpoint /partidos não requer filtro de data — retorna todos os
    partidos com representação parlamentar atual.

    Campos retornados pela listagem:
        id        : identificador único do partido
        sigla     : ex. "PT", "PL", "UNIÃO"
        nome      : nome completo
        uri       : link para detalhes (não vamos usar na carga)

    Retorna
    -------
    list[dict] com todos os partidos
    """
    log.info("=" * 50)
    log.info("EXTRAINDO: /partidos")
    log.info("=" * 50)

    dados = paginar(
        path="/partidos",
        params_extras={"ordem": "ASC", "ordenarPor": "sigla"}
        )

    salvar_raw("partidos", dados)
    return dados
# %%
#Extração `Deputados` - Listagem
def extrair_deputados() -> list[dict]:
    """
    Extrai a listagem de todos os deputados em exercício.

    A listagem retorna campos resumidos. Para campos extras (redeSocial,
    gabinete, etc.) usamos o endpoint de detalhe — veja extrair_deputados_detalhes().

    Campos retornados pela listagem:
        id             : identificador único — CHAVE para outras tabelas
        nome           : nome parlamentar
        siglaPartido   : sigla do partido (relaciona com partidos.sigla)
        siglaUf        : estado que representa
        idLegislatura  : número da legislatura atual
        urlFoto        : URL da foto oficial
        email          : email parlamentar
        uri            : link para o detalhe

    Retorna
    -------
    list[dict] com todos os deputados
    """
    log.info("=" * 50)
    log.info("EXTRAINDO: /deputados (listagem)")
    log.info("=" * 50)

    dados = paginar(
        path="/deputados",
        params_extras={"ordem": "ASC", "ordenarPor": "nome"},
    )

    salvar_raw("deputados", dados)
    return dados
# %%
#Extração `Deputados` - Detalhes
def extrair_deputados_detalhes(deputados: list[dict]) -> list[dict]:
    """
    Para cada deputado, busca o endpoint /deputados/{id} que retorna
    campos que não aparecem na listagem:
        - municipioNascimento, dataNascimento
        - escolaridade
        - situação (em exercício, licenciado, etc.)
        - gabinete: número, andar, telefone

    Por que fazer isso separado?
        A listagem traz os campos suficientes para a maioria das análises.
        O detalhe tem informações complementares que enriquecem a dimensão,
        mas exige 1 chamada por deputado (~513 chamadas extras).
        Separar permite escolher se quer enriquecer ou não.

    Parâmetros
    ----------
    deputados : lista retornada por extrair_deputados()

    Retorna
    -------
    list[dict] com o detalhe de cada deputado
    """
    log.info("=" * 50)
    log.info("EXTRAINDO: /deputados/{id} (detalhes individuais)")
    log.info("Total de deputados: %d", len(deputados))
    log.info("=" * 50)

    detalhes = []

    for i, dep in enumerate(deputados, start=1):
        dep_id = dep.get("id")
        dep_nome = dep.get("nome", "?")

        if not dep_id:
            log.warning("Deputado sem id encontrado: %s - pulando", dep_nome)
            continue

        url = f"{BASE_URL}/deputados/{dep_id}"

        try:
            data = get_com_retry(url, params={})
            detalhe = data.get("dados", {})
            detalhes.append(detalhe)

            if i % 50 == 0 or i == len(deputados):
                log.info("Progresso: %d/%d deputados processados", i, len(deputados))
        
        except Exception as exc:
            #Não interrompe o loop por um deputado que falhou
            log.error("Falha ao buscar detalhe do deputado %s (id=%s): %s", dep_nome, dep_id, exc)

        time.sleep(SLEEP_ENTRE_CHAMADAS)
    
    salvar_raw("deputados_detalhes", detalhes)
    log.info("Detalhes extraídos com sucesso: %d/%d", len(detalhes), len(deputados))
    return detalhes

#%%
#Extração `Proposições`
def extrair_proposicoes(data_inicio: str, data_fim: str) -> list[dict]:
    """
    Extrai proposições apresentadas no período informado.

    A API filtra por dataInicio/dataFim (data de apresentação da proposição).
    O volume pode ser alto — para 30 dias espere entre 300 e 1000 registros
    dependendo do calendário legislativo.

    Campos relevantes:
        id               → chave primária, aparece em votacoes
        siglaTipo        → PL, PEC, MPV, REQ, etc.
        numero           → número da proposição
        ano              → ano de apresentação
        ementa           → texto resumido — BASE para os embeddings de IA
        dataApresentacao → data de entrada na Câmara

    Parâmetros
    ----------
    data_inicio : YYYY-MM-DD
    data_fim    : YYYY-MM-DD
    """

    log.info("=" * 50)
    log.info("EXTRAINDO: /proposicoes | periodo: %s -> %s", data_inicio, data_fim)
    log.info("=" * 50)

    dados = paginar(
        path = "/proposicoes",
        params_extras = {
            "dataInicio": data_inicio,
            "dataFim": data_fim,
            "ordem": "DESC",
            "ordenarPor": "id",
        }
    )

    salvar_raw("proposicoes", dados)
    log.info("Proposições extraídas com sucesso: %d registros", len(dados))
    return dados 

#%%
#Extração `Votações`
def extrair_votacoes(data_inicio: str, data_fim: str) -> list[dict]:
    """
    Extrai votações realizadas em plenário no período informado.

    Campos relevantes:
        id                → chave primária, usada em /votacoes/{id}/votos
        data              → data da votação
        descricao         → descrição do que foi votado
        aprovacao         → 1 (aprovado) ou 0 (rejeitado)
        proposicao_.id    → chave de relacionamento com proposicoes (atenção
                            ao nome aninhado — será achatado no transform)

    Parâmetros
    ----------
    data_inicio : YYYY-MM-DD
    data_fim    : YYYY-MM-DD
    """
    log.info("=" * 50)
    log.info("EXTRAINDO: /votacoes | período: %s → %s", data_inicio, data_fim)
    log.info("=" * 50)

    dados = paginar(
        path="/votacoes",
        params_extras={
            "dataInicio": data_inicio,
            "dataFim":    data_fim,
            "ordem":      "DESC",
            "ordenarPor": "dataHoraRegistro",
        },
    )

    salvar_raw("votacoes", dados)
    log.info("✅ Votações: %d registros", len(dados))
    return dados

#%%
#Extração `Votos`
def extrair_votos(votacoes: list[dict]) -> list[dict]:
    """
    Para cada votação, chama /votacoes/{id}/votos para obter o voto
    individual de cada deputado.

    Por que extrair separado?
        Os votos são o dado mais granular do projeto — é aqui que você
        consegue responder "como fulano vota em temas de tecnologia".
        Mas exige 1 chamada por votação, então separamos para controle.

    Estrutura do registro de voto retornado pela API:
        deputado_.id     → id do deputado (chave de relacionamento)
        deputado_.nome   → nome (redundante, mas útil para debug)
        tipoVoto         → "Sim", "Não", "Abstenção", "Obstrução", etc.
        dataRegistroVoto → timestamp do registro

    O campo 'id_votacao' é adicionado por nós antes de salvar,
    pois a API não o inclui no corpo de resposta dos votos individuais.

    Parâmetros
    ----------
    votacoes : lista retornada por extrair_votacoes()
    """
    log.info("=" * 50)
    log.info("EXTRAINDO: /votacoes/{id}/votos (votos individuais)")
    log.info("Total de votações: %d", len(votacoes))
    log.info("=" * 50)

    todos_votos = []

    for i, vot in enumerate(votacoes, start=1):
        vot_id = vot.get("id")

        if not vot_id:
            log.warning("Votação sem id na posição %d — pulando", i)
            continue

        url = f"{BASE_URL}/votacoes/{vot_id}/votos"

        try:
            data  = get_com_retry(url, params={})
            votos = data.get("dados", [])

            # Injeta o id da votação em cada voto antes de acumular.
            # A API não inclui esse campo no corpo dos votos individuais,
            # mas precisamos dele para o JOIN votacoes ↔ votos no banco.
            for voto in votos:
                voto["id_votacao"] = vot_id

            todos_votos.extend(votos)

            if i % 20 == 0 or i == len(votacoes):
                log.info(
                    "  Progresso: %d/%d votações | %d votos acumulados",
                    i, len(votacoes), len(todos_votos)
                )

        except Exception as exc:
            log.error("Falha ao buscar votos da votação %s: %s", vot_id, exc)

        time.sleep(SLEEP_ENTRE_CHAMADAS)

    salvar_raw("votos", todos_votos)
    log.info("✅ Votos individuais: %d registros", len(todos_votos))
    return todos_votos

# %%
#Entrypoint
def main():
    parser = argparse.ArgumentParser(
        description="Extrai dados da API da Câmara dos Deputados"
    )
    parser.add_argument(
        "--endpoint",
        choices=["partidos", "deputados", "proposicoes", "votacoes", "tudo"],
        default="tudo",
        help="Qual endpoint extrair (padrão: tudo)",
    )
    parser.add_argument(
        "--dias",
        type= int,
        default= 30,
        help= "Quantos dias retroativos extrair para proposições e votações (padrão: 30)"
    )
    parser.add_argument(
        "--com-detalhes",
        action="store_true",
        help="Se presente, também busca o detalhe individual de cada deputado (~513 chamadas extras)",
    )
    parser.add_argument(
        "--com-votos",
        action= "store_true",
        help= "Busca votos individuais de cada votação (1 chamada por votação)"
    ) 
    args = parser.parse_args()
    
    data_inicio, data_fim = calcular_periodo(args.dias)
    log.info("Período: %s -> %s (%d dias)", data_inicio, data_fim, args.dias)

    # Extrai tudo
    #Dimensões
    if args.endpoint in ("partidos", "tudo"):
        partidos = extrair_partidos()
        log.info("✅ Partidos extraídos: %d registros", len(partidos))

    if args.endpoint in ("deputados", "tudo"):
        deputados = extrair_deputados()
        if args.com_detalhes:
            extrair_deputados_detalhes(deputados)
            log.info("✅ Deputados extraídos: %d registros", len(deputados))

    #Fatos
    if args.endpoint in ("proposicoes", "tudo"):
        extrair_proposicoes(data_inicio, data_fim)

    if args.endpoint in ("votacoes", "tudo"):
        votacoes = extrair_votacoes(data_inicio, data_fim)
        if args.com_votos:
            extrair_votos(votacoes)

    log.info("=" * 50)
    log.info("Extração concluída. Arquivos salvos em data/raw/")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
