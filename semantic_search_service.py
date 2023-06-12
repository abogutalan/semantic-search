import os
import openai
import pandas as pd
import elasticsearch

# Define a class for the semantic search service
class SemanticSearchService:

    # Initialize the class with the OpenAI API key and the Elasticsearch client
    def __init__(self, openai_api_key, es_client):
        self.openai_api_key = openai_api_key
        self.es_client = es_client

    # Define a method to get embeddings from OpenAI
    def get_embedding(self, text, model="text-embedding-ada-002"):
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        embeddings = response['data'][0]['embedding']
        return embeddings

    # Define a method to index data with embeddings in Elasticsearch
    def index_data(self, data, index_name):
        # Delete the index if it already exists
        if self.es_client.indices.exists(index=index_name):
            self.es_client.indices.delete(index=index_name)
        # Create the index with a mapping for dense_vector
        self.es_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "Title": {
                            "type": "text"
                        },
                        "Text": {
                            "type": "text"
                        },
                        "Link": {
                            "type": "text"
                        },
                        "Embedding": {
                            "type": "dense_vector",
                            "dims": 1536 # The dimension of OpenAI embeddings
                        }
                    }
                }
            }
        )
        # Index each document with its embedding
        for i, row in data.iterrows():
            text_embedding = self.get_embedding(row["Text"])
            self.es_client.index(
                index=index_name,
                id=i,
                body={
                    "Title": row["Title"],
                    "Text": row["Text"],
                    "Link": row["Link"],
                    "Embedding": text_embedding
                },
            )
        # Refresh the entire index after indexing all the documents
        self.es_client.indices.refresh(index=index_name)

    # Define a method to perform semantic search on the indexed data
    def semantic_search(self, query, index_name, top_k=3):
        # Get the embedding for the query
        query_embedding = self.get_embedding(query)
        # Prepare the search query using cosine similarity script score
        search_query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        # Use cosine similarity as the similarity metric
                        # Add 1.0 to avoid negative scores
                        # Divide by 2.0 to normalize scores between 0 and 1
                        "source": "(cosineSimilarity(params.query_embedding, 'Embedding') + 1.0) / 2.0",
                        # Pass the query embedding as a parameter
                        "params": {
                            "query_embedding": query_embedding
                        }
                    }
                }
            },
            # Return only the top k results
            "size": top_k 
        }
        # Execute the search query
        response = self.es_client.search(
            index=index_name,
            body=search_query
        )
        # Parse the response
        hits = response["hits"]["hits"]
        # Return the results as a list of (score, title, text) tuples
        results = [(hit["_score"], hit["_source"]["Title"], hit["_source"]["Link"],hit["_source"]["Text"]) for hit in hits]
        return results

# Define a function for the main logic of the program
def main():
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
        ],
        "Link": ["https://sparknlp.org/docs/en/annotators#chunk2doc", "https://sparknlp.org/docs/en/annotators#chunkembeddings", "https://sparknlp.org/docs/en/annotators#chunktokenizer", "https://sparknlp.org/docs/en/annotators#textmatcher"]
    }).reset_index()

    # Index the data with embeddings
    index_name = "sample"
    semantic_search_service.index_data(df, index_name)

    # Search the index with a query
    search_text = "How can I match texts?"
    results = semantic_search_service.semantic_search(search_text, index_name)

    # Print the results
    for result in results:
        print(f"Similarity: {result[0]} \nTitle: {result[1]} \nLink: {result[2]} \nText: {result[3]}\n")

# Run the main function
if __name__ == "__main__":
    main()
