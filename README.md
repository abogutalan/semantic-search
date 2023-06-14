
# Semantic Search Service
This project implements a semantic search service using OpenAI and Elasticsearch. It allows users to index a data set of sentences with their embeddings and perform semantic search on them using natural language queries.

### Requirements
* Python 3.10 or higher
* OpenAI API key
* Elasticsearch 8.8 or higher
* pip

### Create python virtual environment and activate
```
python -m venv venv
source venv/bin/activate
```

### Installation
To install the required dependencies, run the following command:

`pip install -r requirements.txt`

This will install the following packages:

* openai
* pandas
* elasticsearch
* requests
* BeautifulSoup


### Docker run:
- Please run the docker command below in a separate terminal.
>docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.8.0

### OpenAI API Key
>export OPENAI_API_KEY=<your_openai_api_key>

### Code run:
>python spark_nlp_scraper.py
>python indexer.py 
>python semantic_search_service.py


### Usage
##### SparkNLPAnnotatorScraper

The SparkNLPAnnotatorScraper class is used to scrape the SparkNLP Annotator documentation and extract the title, text, and link of each section.
You can use it like this:
```
from SparkNLPAnnotatorScraper import SparkNLPAnnotatorScraper

# Use the class to scrape the website and save the data to a CSV
scraper = SparkNLPAnnotatorScraper()
scraper.scrape_website()
scraper.to_csv('annotators.csv')
```

##### To use the semantic search service, you need to follow these steps:

1. Create an instance of the SemanticSearchService class by passing your OpenAI API key and Elasticsearch client as arguments.
2. Call the index_data method with your data set (a pandas DataFrame with Title and Text columns) and an index name as arguments. This will index your data set with embeddings in Elasticsearch.
3. Call the semantic_search method with your query (a natural language string), the index name, and an optional top_k argument (the number of results to return) as arguments. This will perform semantic search on your indexed data and return a list of (score, title, text) tuples.
For example:

```
import os
import pandas as pd
import elasticsearch
from semantic_search_service import SemanticSearchService

# Set up OpenAI API key and Elasticsearch client
openai_api_key = os.environ.get("OPENAI_API_KEY")
es_client = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])

# Create an instance of the semantic search service class
semantic_search_service = SemanticSearchService(openai_api_key, es_client)

# Define a simple sample data set of sentences
df = pd.DataFrame({
    "Title": ["Chunk2Doc", "ChunkEmbeddings", "ChunkTokenizer", "TextMatcher"],
    "Text": [
        "Converts a CHUNK type column back into DOCUMENT. Useful when trying to re-tokenize or do further analysis on a CHUNK result.",
        "This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate chunk embeddings from either Chunker, NGramGenerator, or NerConverter outputs.For extended examples of usage, see the Examples and the ChunkEmbeddingsTestSpec.",
        "Tokenizes and flattens extracted NER chunks.The ChunkTokenizer will split the extracted NER CHUNK type Annotations and will create TOKEN type Annotations. The result is then flattened, resulting in a single array.For extended examples of usage, see the ChunkTokenizerTestSpec.",
        "Annotator to match exact phrases (by token) provided in a file against a Document. A text file of predefined phrases must be provided with setEntities. For extended examples of usage, see the Examples and the TextMatcherTestSpec."
    ]
}).reset_index()

# Index the data with embeddings
index_name = "sample"
semantic_search_service.index_data(df, index_name)

# Search the index with a query
search_text = "How can I match texts?"
results = semantic_search_service.semantic_search(search_text, index_name)

# Print the results
for result in results:
    print(f"Similarity: {result[0]} \nTitle: {result[1]} \nText: {result[2]}\n")
```

### Input:
```search_text = "How to convert token to chunk?"```

### Output:
```
Similarity: 0.913725 
Title: token2chunk 
Link: https://sparknlp.org/docs/en/annotators#token2chunk

Similarity: 0.89781594 
Title: stopwordscleaner 
Link: https://sparknlp.org/docs/en/annotators#stopwordscleaner

Similarity: 0.89720714 
Title: chunktokenizer 
Link: https://sparknlp.org/docs/en/annotators#chunktokenizer

Similarity: 0.89308345 
Title: chunk2doc 
Link: https://sparknlp.org/docs/en/annotators#chunk2doc

Similarity: 0.88517475 
Title: nerconverter 
Link: https://sparknlp.org/docs/en/annotators#nerconverter

```

### Testing
To run the unittest for this project, run the following command:

`python -m unittest test_semantic_search_service.py`

To generate a coverage report for this project, run the following commands:

```
coverage run test_semantic_search_service.py
coverage report
```

This will print a summary of the coverage report to the standard output, showing the name, statements, missed statements, and percentage of each module. For example:
```
Name                              Stmts   Miss  Cover
-----------------------------------------------------
semantic_search_service.py           40     12    70%
test_semantic_search_service.py      40      0   100%
-----------------------------------------------------
TOTAL                                80     12    85%
```
