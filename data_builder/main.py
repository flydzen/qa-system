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

try:
    client.drop_database('qa')
except exceptions.MilvusException as e:
    print(e)
client.create_database('qa')

article_max_len = 20480
schema = CollectionSchema(
    [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=384, description='Embedding of the text'),
        FieldSchema(name='topic', dtype=DataType.VARCHAR, max_length=100, description='Topic', is_partition_key=True),
        FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=article_max_len, description='Article')
    ]
)
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name='embedding',
    metric_type='IP',
    index_type='IVF_FLAT',
    index_name='embedding_index',
    params={'nlist': 128}
)

collection_name = 'articles'
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

collection = client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
)

###

df = pd.read_csv('Articles.csv', encoding='cp1252')[['Article', 'NewsType']]
df['Article'] = df['Article'].apply(lambda x: x[:article_max_len].strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = milvus_model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device=device,
    normalize_embeddings=True,
)

embeddings = []
batch_size = 64
# embeddings = model.encode_documents(df['Article'].to_list())
for i in tqdm(list(range(0, len(df), batch_size))):
    batch = df.iloc[i:i + batch_size]
    batch_embeddings = model.encode_documents(batch['Article'].to_list())
    embeddings.extend(batch_embeddings)

insert_df = pd.DataFrame({
    'embedding': embeddings,
    'topic': df['NewsType'].iloc[:len(embeddings)],
    'text': df['Article'].iloc[:len(embeddings)],
})
data_formatted = insert_df.to_dict('records')
client.insert(
    collection_name=collection_name,
    data=data_formatted,
)
client.load_collection(collection_name)
