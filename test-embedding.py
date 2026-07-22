import ollama
import numpy as np

# Create explicit client
client = ollama.Client(host="http://127.0.0.1:11434")


def get_embedding(text):
    response = client.embed(
        model="mxbai-embed-large:latest",
        input=text
    )
    return response["embeddings"][0]


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


sentence1 = "Hitler"        #cat
sentence2 = "Italy"         #bark
sentence3 = "Germany"       #meow
sentence4 = "Mussolini"     #dog

embedding1 = get_embedding(sentence1)
embedding2 = get_embedding(sentence2)
embedding3 = get_embedding(sentence3) 
embedding4 = get_embedding(sentence4)

print("Embedding 1 length:", len(embedding1))
print("\nEmbedding 1:")
print(embedding1[:10])


print("Embedding 2 length:", len(embedding2))
print("\nEmbedding 2:")
print(embedding2[:10])

print("Embedding 3 length:", len(embedding3))
print("\nEmbedding 3:")
print(embedding3[:10])

print("Embedding 4 length:", len(embedding4))
print("\nEmbedding 4:")
print(embedding4[:10])

similarity12 = cosine_similarity(embedding1, embedding2)
print("Cosine Similarity 1 & 2:", similarity12)

similarity13 = cosine_similarity(embedding1, embedding3)
print("Cosine Similarity 1 & 3:", similarity13)

similarity14 = cosine_similarity(embedding1, embedding4)
print("Cosine Similarity 1 & 4:", similarity14)

similarity23 = cosine_similarity(embedding2, embedding3)
print("Cosine Similarity 2 & 3:", similarity23)

similarity24 = cosine_similarity(embedding2, embedding4)
print("Cosine Similarity 2 & 4:", similarity24)

similarity34 = cosine_similarity(embedding3, embedding4)
print("Cosine Similarity 3 & 4:", similarity34)


# Convert to numpy arrays
e1 = np.array(embedding1)
e2 = np.array(embedding2)
e3 = np.array(embedding3)

# Perform vector arithmetic
result = e1 + e2 - e3

print("Result vector:", result)

print("\nEmbedding Result:")
print(result[:10])

similarity = cosine_similarity(result, embedding4)
print("Cosine Similarity: Result & Embedding 4:", similarity)