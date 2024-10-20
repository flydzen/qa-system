import os

import httpx
import pandas as pd
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient, exceptions
from tqdm import tqdm

milvus_host = os.getenv('MILVUS_HOST') or 'localhost'
milvus_port = os.getenv('MILVUS_PORT') or '19530'
llm_host = os.getenv('LLM_HOST') or 'localhost'
llm_port = os.getenv('LLM_PORT') or '8080'

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

url = f'http://{llm_host}:{llm_port}/encode'
embeddings = []
batch_size = 64
for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i + batch_size]
    articles = batch['Article'].to_list()
    response = httpx.post(url, json={'items': articles})

    if response.status_code == 200:
        embeddings.extend(response.json())
    else:
        print(f"Error with status code: {response.status_code}")

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
