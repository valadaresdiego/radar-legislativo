#%%
from pathlib import Path

BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"

REQUEST_TIMEOUT = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_ROOT / "data"
EXPLORATION_PATH = DATA_PATH / "exploration"
