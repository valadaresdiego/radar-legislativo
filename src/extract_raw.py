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
        (sem flag)              → extrai ambos

Saída:
    data/raw/partidos_YYYY-MM-DD.json
    data/raw/deputados_YYYY-MM-DD.json
    data/raw/deputados_detalhes_YYYY-MM-DD.json  (campos extras de cada deputado)
"""

import argparse 
import logging 
import time 

from camara_client import(
    BASE_URL,
    SLEEP_ENTRE_CHAMADAS,
    get_com_retry,
    paginar,
    salvar_raw,
)

log = logging.getLogger(__name__)
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
# %%
#Entrypoint
def main():
    parser = argparse.ArgumentParser(
        description="Extrai partidos e/ou deputados da API da Câmara."
    )
    parser.add_argument(
        "--endpoint",
        choices=["partidos", "deputados", "ambos"],
        default="ambos",
        help="Qual endpoint extrair (padrão: ambos)",
    )
    parser.add_argument(
        "--com-detalhes",
        action="store_true",
        help="Se presente, também busca o detalhe individual de cada deputado (~513 chamadas extras)",
    ) 
    args = parser.parse_args()
    
    # Executa conforme a flag escolhida
    if args.endpoint in ("partidos", "ambos"):
        partidos = extrair_partidos()
        log.info("✅ Partidos extraídos: %d registros", len(partidos))

    if args.endpoint in ("deputados", "ambos"):
        deputados = extrair_deputados()
        log.info("✅ Deputados extraídos: %d registros", len(deputados))

        if args.com_detalhes:
            detalhes = extrair_deputados_detalhes(deputados)
            log.info("✅ Detalhes de deputados extraídos: %d registros", len(detalhes))

    log.info("=" * 50)
    log.info("Extração concluída. Arquivos salvos em data/raw/")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
