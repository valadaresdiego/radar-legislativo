-- Decisão de FK:
--   Não usamos FOREIGN KEY constraints neste schema inicial.
--   Motivo: a extração filtra por janela de 30 dias, então uma votação pode
--   referenciar uma proposição que ainda não está na nossa base.
--   A integridade referencial é garantida pelo pipeline de extração.
--   FKs podem ser adicionadas depois de uma carga histórica completa.
--
-- Ordem de execução importa:
--   Dimensões primeiro (partidos, deputados), fatos depois.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- DIMENSÃO: partidos
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS partidos (
    id            INTEGER     PRIMARY KEY,
    sigla         TEXT        NOT NULL,
    nome          TEXT        NOT NULL,
    url_logo      TEXT,
    data_extracao TIMESTAMP
);

COMMENT ON TABLE  partidos             IS 'Partidos com representação parlamentar atual';
COMMENT ON COLUMN partidos.url_logo    IS 'URI da API — não é a URL do logo. Enriquecer via /partidos/{id} se necessário.';
COMMENT ON COLUMN partidos.data_extracao IS 'Timestamp de quando o dado foi capturado da API';


-- ---------------------------------------------------------------------------
-- DIMENSÃO: deputados
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deputados (
    id            INTEGER     PRIMARY KEY,
    nome          TEXT        NOT NULL,
    sigla_partido TEXT,
    uf            TEXT,
    email         TEXT,
    url_foto      TEXT,
    data_extracao TIMESTAMP
);

COMMENT ON TABLE  deputados                IS 'Deputados federais em exercício na legislatura atual';
COMMENT ON COLUMN deputados.sigla_partido  IS 'Sigla do partido — relaciona com partidos.sigla';
COMMENT ON COLUMN deputados.uf            IS 'Estado representado (siglaUf na API)';


-- ---------------------------------------------------------------------------
-- FATO: proposicoes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS proposicoes (
    id                INTEGER     PRIMARY KEY,
    tipo              TEXT,
    numero            INTEGER,
    ano               INTEGER,
    ementa            TEXT,
    data_apresentacao DATE,
    tema              TEXT,        -- preenchido pela camada de IA (Etapa 4)
    resumo_executivo  TEXT,        -- preenchido pela camada de IA (Etapa 4)
    data_extracao     TIMESTAMP
);

COMMENT ON TABLE  proposicoes                  IS 'Proposições legislativas (PLs, PECs, MPVs, etc.)';
COMMENT ON COLUMN proposicoes.tipo             IS 'Tipo da proposição: PL, PEC, MPV, REQ, etc.';
COMMENT ON COLUMN proposicoes.ementa           IS 'Texto da ementa — campo base para embeddings de IA';
COMMENT ON COLUMN proposicoes.tema             IS 'Classificação temática gerada por embeddings (Etapa 4)';
COMMENT ON COLUMN proposicoes.resumo_executivo IS 'Resumo gerado por LLM para consumo executivo (Etapa 4)';


-- ---------------------------------------------------------------------------
-- FATO: votacoes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS votacoes (
    id              TEXT        PRIMARY KEY,   -- API retorna strings, não ints
    id_proposicao   INTEGER,                   -- FK lógica → proposicoes.id
    data            DATE,
    descricao       TEXT,
    aprovada        BOOLEAN,
    data_extracao   TIMESTAMP
);

COMMENT ON TABLE  votacoes               IS 'Votações realizadas em plenário';
COMMENT ON COLUMN votacoes.id            IS 'ID no formato string da API (ex: "abc123")';
COMMENT ON COLUMN votacoes.id_proposicao IS 'Referência à proposição votada (sem FK constraint — ver nota no topo)';
COMMENT ON COLUMN votacoes.aprovada      IS 'TRUE se aprovada, FALSE se rejeitada';


-- ---------------------------------------------------------------------------
-- FATO: votos
-- Chave primária composta: (id_votacao, id_deputado)
-- Um deputado tem exatamente um voto por votação.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS votos (
    id_votacao    TEXT        NOT NULL,        -- FK lógica → votacoes.id
    id_deputado   INTEGER     NOT NULL,        -- FK lógica → deputados.id
    voto          TEXT,                        -- 'Sim', 'Não', 'Abstenção', 'Obstrução'
    data_extracao TIMESTAMP,
    PRIMARY KEY (id_votacao, id_deputado)
);

COMMENT ON TABLE votos IS 'Votos individuais de cada deputado em cada votação';
COMMENT ON COLUMN votos.voto IS 'Valores possíveis: Sim, Não, Abstenção, Obstrução, Art. 17, Presidência';


-- ---------------------------------------------------------------------------
-- ÍNDICES — aceleram as queries mais comuns do produto
-- ---------------------------------------------------------------------------

-- Busca de proposições por período (query mais frequente do produto)
CREATE INDEX IF NOT EXISTS idx_proposicoes_data
    ON proposicoes (data_apresentacao DESC);

-- Busca de proposições por tema (após camada de IA)
CREATE INDEX IF NOT EXISTS idx_proposicoes_tema
    ON proposicoes (tema)
    WHERE tema IS NOT NULL;

-- Busca de votações por data
CREATE INDEX IF NOT EXISTS idx_votacoes_data
    ON votacoes (data DESC);

-- Busca de todos os votos de um deputado
CREATE INDEX IF NOT EXISTS idx_votos_deputado
    ON votos (id_deputado);

-- Busca de todos os votos de uma votação
CREATE INDEX IF NOT EXISTS idx_votos_votacao
    ON votos (id_votacao);


-- ---------------------------------------------------------------------------
-- Verificação final — deve retornar 5 tabelas
-- ---------------------------------------------------------------------------
SELECT table_name, obj_description(
    (quote_ident(table_schema) || '.' || quote_ident(table_name))::regclass
) AS descricao
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('partidos', 'deputados', 'proposicoes', 'votacoes', 'votos')
ORDER BY table_name;
