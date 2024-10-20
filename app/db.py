import os

from pymilvus import MilvusClient

from app.models import DBSearchResponse

milvus_host = os.getenv('MILVUS_HOST') or 'localhost'
milvus_port = os.getenv('MILVUS_PORT') or '19530'


class Database:
    COLLECTION_NAME = 'articles'

    def __init__(self, host=milvus_host, port=milvus_port):
        self.client = MilvusClient(uri=f'http://{host}:{port}', timeout=10)

    def search(self, topic, embeddings: list) -> list[DBSearchResponse]:
        res = self.client.search(
            collection_name=self.COLLECTION_NAME,
            data=embeddings,
            anns_field='embedding',
            limit=3,
            search_params={'metric_type': 'IP', 'params': {}},
            output_fields=['topic', 'text'],
            filter=f'topic == "{topic}"',
        )
        return [
            DBSearchResponse(items=result)
            for result in res
        ]
