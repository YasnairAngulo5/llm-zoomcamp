# Vector Search 
Vector search is a method of finding similar items in a dataset by comparing their vector representations. Unlike traditional keyword-based search, which relies on exact matches, vector search uses mathematical representations of data to find items that are similar in meaning or context.

## Why to use vector search?
- Semantic Understanding: Vector search can understand the context and meaning behind words, making it more effective for natural language processing tasks.
- Scalability: It can handle large datasets efficiently.
- Flexibility: It can be used for various types of data, including text, images, and audio.

## How Does Vector Search Work?
1. **Data Representation:** Data is converted into vectors using techniques like word embeddings, sentence embeddings, document embeddings, or image embeddings.
2. **Indexing:** These vectors are stored in a vector database or an index.
3. **Querying:** When a query is made, it is also converted into a vector.
4. **Similarity Measurement:** The query vector is compared with the indexed vectors using similarity measures like cosine similarity or Euclidean distance.
5. **Retrieval:** The most similar items are retrieved and presented as search results.

### Key Concepts

#### What are Embeddings?

Embeddings are dense vector representations of data that capture the semantic meaning and associations of text, images, and other data types, placing similar meanings closer together in vector space.
They are fundamental to vector search as they allow for the comparison of data in a continuous vector space. Various types of data, including words, sentences, documents, and images, can be converted into embeddings. 

#### How are Embeddings Generated?

The process of creating vector embeddings involves several steps:
1. **Data Collection and Preprocessing**: Gathering and preparing the data.
2. **Training the Model**: Using machine learning to identify patterns and representations in the data.
3. **Generating Embeddings**: Creating numerical representations based on the trained model.
4. **Iteration**: Continuously refining the model to improve the quality of embeddings.

Pre-trained models from platforms like Hugging Face can be leveraged to simplify this process. For sentence embeddings, models like Sentence Transformers (e.g., 'all-mpnet-base-v2') are commonly used. These models are trained to capture the context and meaning of sentences by converting them into high-dimensional vectors.


#### The Role of Indexing in Vector Databases

Efficient storage and retrieval of data are critical for vector databases. Indexing methods ensure that data is managed optimally, allowing for fast and accurate searches. Different indexing techniques are used to achieve this efficiency.

#### Similarity Measures

- **Cosine Similarity**: Measures the cosine of the angle between two vectors. It ranges from -1 to 1, where 1 means the vectors are identical.
- **Euclidean Distance**: Measures the straight-line distance between two vectors in a multi-dimensional space.

## Semantic Search with Elasticsearch

### Overview

Here we will learn how to create a semantic search engine using Elasticsearch and Python. Semantic search improves traditional search by understanding the meaning behind the search terms, providing more relevant results. In this project, we’ll use Elasticsearch to handle the vector search, allowing us to implement semantic search with ease.

### Why Use Elasticsearch for Semantic Search?
- Scalable: Elasticsearch can manage large datasets and handle numerous search queries at once.
- Flexible: It supports various types of data, such as text, numbers, and even geospatial information.
- Advanced features: Elasticsearch offers powerful search capabilities, including full-text search and filtering.

#### Key Concepts

##### Documents

In Elasticsearch, data is stored in documents, which are like JSON objects containing fields and values. For example, a document could represent a book with fields like “title,” “author,” and “published date.”

##### Indexes

An index is like a collection of documents, optimized for searching. In a way, it’s similar to a table in a relational database, but it’s designed to be more flexible and capable of handling complex data types.

## Setting Up Your Environment

To get started, you’ll need Docker to run Elasticsearch, and a pre-trained model to generate embeddings. First, ensure Docker is running and then execute this command to start an Elasticsearch instance:
```bash
    docker run -it \
        --rm \
        --name elasticsearch \
        -m 4GB \
        -p 9200:9200 \
        -p 9300:9300 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```
If the previous command doesn't work (i.e. you see "error pulling image configuration"), try to run ElasticSearch directly from Docker Hub:
    
```bash
    docker run -it \
        --rm \
        --name elasticsearch \
        -p 9200:9200 \
        -p 9300:9300 \
        -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" \
        elasticsearch:8.4.3
```
You can check if your elastic instance is running by sending a curl request to it: curl localhost:9200.

## Loading and Preprocessing Data

Next, you’ll load and preprocess the data. We will use the `documents.json` file containing the list of documents. We will extract each document and add a course field to indicate which course it belongs to

## Embeddings with Sentence Transformers

To perform semantic search, we convert text data into vector representations, called embeddings, that capture the meaning of the text. We’ll use a pre-trained model from the sentence-transformers library to generate these embeddings. These embeddings are then stored in Elasticsearch.

**Steps:**

1.	Install the `sentence_transformers` library.

    ```bash
        pip install sentence_transformers==2.7.0
    ````

2.	Load the pre-trained model to create embeddings for your documents.
```python
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-mpnet-base-v2")
```
## Connecting to Elasticsearch

Once Elasticsearch is running, use Python to establish a connection. Here’s a simple example using the elasticsearch package:

```python
    from elasticsearch import Elasticsearch

    # Connect to the Elasticsearch instance
    es_client = Elasticsearch('http://localhost:9200')

    # Check the connection
    print(es_client.info())
```

## Creating an Index and Defining Mappings

In Elasticsearch, mapping defines the structure of your documents. You need to specify the data types for each field and how Elasticsearch should index them.

Here’s an example of an index with settings for storing embeddings (dense vectors) that will allow you to perform semantic search:

```pyhton
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}
            }
        }
    }

    index_name = "course-questions"

    # Delete the index if it exists
    es_client.indices.delete(index=index_name, ignore_unavailable=True)

    # Create the index
    es_client.indices.create(index=index_name, body=index_settings)
```

## Indexing Documents

Now, you need to index your preprocessed documents, along with the embeddings generated from the model, into Elasticsearch:

```python
    for doc in operations:
        try:
            es_client.index(index=index_name, document=doc)
        except Exception as e:
            print(e)
```

## Querying with Semantic Search

To perform a semantic search, convert the user’s query into embeddings and search within Elasticsearch. Here’s an example:

```python
    search_term = "windows or mac?"
    vector_search_term = model.encode(search_term)

    query = {
        "field": "text_vector",
        "query_vector": vector_search_term,
        "k": 5,
        "num_candidates": 10000
    }

    res = es_client.search(index=index_name, knn=query, source=["text", "section", "question", "course"])
    res["hits"]["hits"]
```

## Hybrid Search (Combining Semantic and Keyword Search)

You can combine semantic and keyword searches for more refined results. This allows you to filter results based on specific criteria while still leveraging the power of semantic search.

```python
knn_query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000
}

response = es_client.search(
    index=index_name,
    query={
        "match": {"section": "General course-related questions"}
    },
    knn=knn_query,
    size=5
)

response["hits"]["hits"]
```

## Understanding the Results

When a search is performed, each result is scored based on its relevance. A higher score (closer to 1) means a better match. You can also explain how scores are calculated by adding the explain=true keyword to your query.

 