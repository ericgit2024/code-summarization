from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def get_encoder_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Loads and returns the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def encode_codes(code_list, model):
    """
    Encodes a list of code strings into embeddings.
    """
    return model.encode(code_list, convert_to_numpy=True)

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from the given embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_similar_codes(query_code, index, dataset, model, k=3):
    """
    Retrieves the top-k most similar codes and their docstrings.
    """
    query_embedding = encode_codes([query_code], model)
    distances, indices = index.search(query_embedding, k)

    retrieved_indices = indices[0].tolist()

    retrieved_codes = [dataset['train'][i]['code'] for i in retrieved_indices]
    retrieved_docstrings = [dataset['train'][i]['docstring'] for i in retrieved_indices]

    return retrieved_codes, retrieved_docstrings, distances[0]

if __name__ == '__main__':
    from data_loader import load_and_clean_dataset

    dataset = load_and_clean_dataset()
    encoder_model = get_encoder_model()

    train_codes = dataset['train']['code']
    train_embeddings = encode_codes(train_codes, encoder_model)

    index = build_faiss_index(train_embeddings)

    sample_query_code = dataset['train'][10]['code']
    retrieved_codes, retrieved_docstrings, distances = retrieve_similar_codes(sample_query_code, index, dataset, encoder_model, k=3)

    print("Sample Query Code:")
    print(sample_query_code)
    print("\nRetrieved Similar Codes and Docstrings:")
    for i in range(len(retrieved_codes)):
        print(f"\n--- Result {i+1} (Distance: {distances[i]:.4f}) ---")
        print("Code:")
        print(retrieved_codes[i])
        print("\nDocstring:")
        print(retrieved_docstrings[i])
