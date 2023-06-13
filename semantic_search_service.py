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

    # # Define a method to get embeddings from OpenAI
    # def get_embedding(self, text, model="text-embedding-ada-002"):
    #     response = openai.Embedding.create(
    #         input=text,
    #         model=model
    #     )
    #     embeddings = response['data'][0]['embedding']
    #     return embeddings

    # Calculate the embeddings by averaging since the model's maximum context length is 8191 tokens.
    def get_embedding(self, text, model="text-embedding-ada-002"):
        max_tokens = 8000  # safe number to avoid hitting the limit
        text_parts = [text[i: i + max_tokens] for i in range(0, len(text), max_tokens)]
        embeddings = []
        for part in text_parts:
            response = openai.Embedding.create(
                input=part,
                model=model
            )
            embeddings.append(response['data'][0]['embedding'])
        # Calculate the average embedding
        avg_embedding = [sum(col) / len(col) for col in zip(*embeddings)]
        return avg_embedding


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
    def semantic_search(self, query, index_name, top_k=5):
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
    # Check if the API key is present
    if not openai_api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it before running the script.")
    
    # Create an instance of the Elasticsearch client
    es_client = elasticsearch.Elasticsearch(hosts=["http://localhost:9200"])

    # Create an instance of the semantic search service class
    semantic_search_service = SemanticSearchService(openai_api_key, es_client)

    # Read the data from a CSV file
    df = pd.read_csv('annotators.csv')

    # Index the data with embeddings
    index_name = "sample"
    semantic_search_service.index_data(df, index_name)

    # Search the index with a query
    search_text = "How to convert token to chunk?"
    results = semantic_search_service.semantic_search(search_text, index_name)

    print("\nWriting to output.txt file...")
    # Print the results
    for result in results:
        # write to a file 
        with open('output.txt', 'a') as f:
            f.write(f"Similarity: {result[0]} \nTitle: {result[1]} \nLink: {result[2]}\n\n") # \nText: {result[3]}\n")

    print("Completed!")
    
# Run the main function
if __name__ == "__main__":
    main()
