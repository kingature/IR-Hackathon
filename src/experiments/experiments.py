from layout import Layout
from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client

tira = Client()
ensure_pyterrier_is_loaded()


class ChainOfThoughts(Layout):
    pass


class SimilarQueriesFS(Layout):
    pass


class SimilarQueriesZS(Layout):
    pass
