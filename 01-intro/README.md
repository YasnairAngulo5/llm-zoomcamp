- [Module 1: Introduction](#module-1-introduction)
  - [Introduction to LLM and RAG](#introduction-to-llm-and-rag)
    - [Large Language Models (LLM)](#large-language-models-llm)
    - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Prearing the Enviroment](#prearing-the-enviroment)
    - [Steps:](#steps)
  - [Search and Retrieval](#search-and-retrieval)
    - [MinSearch package](#minsearch-package)
    - [Elasticsearch python library](#elasticsearch-python-library)
    - [Generation with OpenAI](#generation-with-openai)
      - [1. How to set and use OpenAI](#1-how-to-set-and-use-openai)
      - [2. Building the prompt](#2-building-the-prompt)
        - [2.1 Prompt template](#21-prompt-template)
        - [2.1 Creating context based on the retrieved results.](#21-creating-context-based-on-the-retrieved-results)
        - [2.2 Setting the *question* and *context* in the template.](#22-setting-the-question-and-context-in-the-template)
        - [2.3 Getting the final answer.](#23-getting-the-final-answer)
  - [Extra materials](#extra-materials)



# Module 1: Introduction
 
In this module, we will learn what LLM and RAG are and implement a simple RAG pipeline to answer questions about 
the FAQ Documents from the Zoomcamp courses hosted by [DataTalksClub](https://datatalks.club/).

* Index Zoomcamp FAQ documents
    * DE Zoomcamp: https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit
    * ML Zoomcamp: https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit
    * MLOps Zoomcamp: https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit

## Introduction to LLM and RAG
### Large Language Models (LLM)
To begin, it's important to understand what language models are. Language Models (LMs) are algorithms designed to predict the next word in a sequence based on the words that have already been provided. A familiar example of this is the predictive text function found on mobile phone keyboards. As you start typing, the keyboard offers suggestions for possible next words to assist you in completing your sentence.

**Large Language Models (LLMs)** are advanced versions of language models. While standard language models predict the next word based on previous words, LLMs are much larger and more powerful because they are trained on massive amounts of text data. This allows them to understand context, nuances, and even generate highly coherent and human-like text. LLMs can handle complex tasks such as answering questions, summarizing documents, translating languages, and more, making them a core part of many AI applications today. Their size and sophistication give them the ability to perform well across various language-related tasks. Some examples are GPT or BERT.

### Retrieval-Augmented Generation (RAG)
RAG enhances LLMs by combining them with information retrieval systems. RAG leverages external data sources, such as databases or documents, to retrieve relevant information in response to user queries. It ensures that responses are more accurate and factually grounded by using up-to-date and domain-specific knowledge, making it particularly useful for tasks where precision is critical.

A simple example of a RAG architecture:
![RAG](./images/RAG.jpg)


## Prearing the Enviroment
To set up the development environment for this project, we utilized GitHub Codespaces. It allows for a cloud-based development environment that runs directly in Visual Studio Code without needing local setup. Follow the steps below to get started:

### Steps:
1. **Open the repository in GitHub Codespaces:**

- Navigate to your repository on GitHub.
- Click the Code button and select Codespaces.
- Choose an existing codespace or create a new one.

2. Environment Setup:

- Once the codespace is open, all dependencies defined in the repository (e.g., devcontainer.json or Dockerfile) will be automatically installed.
- Install the following libraries:
    ```bash
    pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn ipywidgets
    ```
## Search and Retrieval
Here, we will learn how to retrieve documents with relevant information and context to provide the appropriate answer using two different search methods.

- MinSearch package
- Elasticsearch python library

Before starting with the search methods, let’s import the FAQ documents (already parsed into JSON) into a list called documents.

1. Run `parse-faq.ipynb` to download FAQ docs and store them as `documents.json` file.

2. Import the document.

 ```python
    import json

    with open('documents.json', 'rt') as f_in:
        docs_raw = json.load(f_in)

    documents = []

    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)
```

### MinSearch package

The search algorithm already implemented is provided by [DataTalksClub](https://datatalks.club/) in `minsearch.py` script , which uses TF-IDF and cosine similarity to retrieve relevant information. The idea behind this option is to have a small, in-memory solution for running our RAG.

To test it in jupyter notebook, let's follow these steps:

1.  Get and Import the package
    ```python
    !wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/01-intro/minsearch.py
    ```
    ```python
    import minsearch
    ``` 
2. Create an index based on the fields in our FAQ document
    ```python
        index = minsearch.Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    ```
3. Get the results
    ```python
    query = 'the course has already started, can I still enroll?'
    ```
    ```python
        boost = {'question':3.0, 'section':0.5}

        results = index.search(
            query=q,
            filter_dict={'course': 'data-engineering-zoomcamp'},
            boost_dict=boost,
            num_results=5
        )
    ```

### Elasticsearch python library
1. Running it with docker
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

2. Create the instance.
    ```python
    from elasticsearch import Elasticsearch
    ```
    ```pyhton
    es_client = Elasticsearch('http://localhost:9200')
    ```
3. Create the index based on the fields in the documents that you want to index.
    ```python
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
                "course": {"type": "keyword"} 
            }
        }
    }

    index_name = "course-questions"

    es_client.indices.create(index=index_name, body=index_settings)
    ```
4. Build the query.
    ```python
    query = 'the course has already started, can I still enroll?'
    ```
    ```python
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    ```

    5. Get the results.
    ```python
    response = es_client.search(index=index_name, body=search_query)
    ```
    ```python
    results_docs = []
    for hit in response['hits']['hits']:
        results_docs.append(hit['_source'])
    ````

At this point, the result for both options should look something like this:
```
[{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, the slack channel remains open and you can ask questions there. But always sDocker containers exit code w search the channel first and second, check the FAQ (this document), most likely all your questions are already answered here.\nYou can also tag the bot @ZoomcampQABot to help you conduct the search, but don’t rely on its answers 100%, it is pretty good though.',
  'section': 'General course-related questions',
  'question': 'Course - Can I get support if I take the course in the self-paced mode?',
  'course': 'data-engineering-zoomcamp'}]
```

### Generation with OpenAI
Now it's time to generate the final asnwer using OpenAI.

#### 1. How to set and use OpenAI

To get started, let's first understand how to use the OpenAI API. Follow these steps to set it up:

- Create an OpenAI account:

    Go to the [OpenAI website](https://beta.openai.com/signup/)  and sign up for an account if you haven't already.
- Obtain an API key:

    Once logged in, navigate to the API section to generate your API key. You'll need this key to authenticate requests to the OpenAI API.
- Export your API key:
    ```bash
    export OPENAI_API_KEY="YOUR-API-KEY"
    ```

- In Jupyter Notebook or any other tool of your preference, run this code:
    ```python
    import openai
    from openai import OpenAI

    response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role":"user", "content": "is it too late to join the course?"}]
    )

    response.choices[0].message.content
    ```

#### 2. Building the prompt
Creating a prompt is important in a RAG system because it helps the model find the right information and give accurate answers based on what the user asks.

##### 2.1 Prompt template
```python
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
Use only the facts from the CONTEXT when answering the QUESTION.
If the CONTEXT doesn't contain answer, output NONE

QUESTION: {question}

CONTEXT:
{context}


""".strip()
```

##### 2.1 Creating context based on the retrieved results.
```python
context = ""

for doc in results:
    context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
```

##### 2.2 Setting the *question* and *context* in the template.
```python
prompt = prompt_template.format(question=query, context=context).strip()
```

##### 2.3 Getting the final answer.
```python
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{"role": "user", "content": prompt}]
)

response.choices[0].message.content
```

We use `"type": "best_fields"`. You can read more about 
different types of `multi_match` search in [elastic-search.md](elastic-search.md).


## Extra materials

* If you're curious to know how the code for parsing the FAQ works, check [this video](https://www.loom.com/share/ff54d898188b402d880dbea2a7cb8064)
