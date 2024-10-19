import json

import milvus_model
import pandas as pd
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient, exceptions
import os
from tqdm import tqdm
import torch

milvus_host = os.getenv('MILVUS_HOST') or 'localhost'
milvus_port = os.getenv('MILVUS_PORT') or '19530'

client = MilvusClient(uri=f'http://{milvus_host}:{milvus_port}')
collection_name = 'articles'
client.load_collection(collection_name)


model = milvus_model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device='cpu',
    normalize_embeddings=True,
)
encodings = model.encode_documents(['Bank of Pakistan (SBP)'])
print(encodings)
res = client.search(
    collection_name=collection_name,
    data=encodings,
    anns_field='embedding',
    limit=5,
    search_params={'metric_type': 'IP', 'params': {}},
    output_fields=['topic', 'text'],
    filter=f'topic == "business"',
)
print(json.dumps(res, indent=4))
