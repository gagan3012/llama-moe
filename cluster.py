from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from datasets import load_dataset
import numpy as np
import json

num_clusters = 16
# Initialize the Sentence Transformer model
model = SentenceTransformer('models/all-MiniLM-L6-v2', device='cuda')

def sentence(s):
    s["sentence"] = s["instruction"] + s["output"]
    return s

def embed(examples):
    sentences = examples['sentence']
    examples['embeddings'] = model.encode(sentences)
    return examples


# Load a sample dataset from Hugging Face
dataset = load_dataset(
    'dataset/WizardLM_evol_instruct_70k', split='train')

# Add embeddings to the dataset
dataset = dataset.map(sentence)

dataset = dataset.map(embed, batched=True, batch_size=1000)

# Collect all embeddings into a list
all_embeddings = [item for sublist in dataset['embeddings'] for item in sublist]

all_embeddings = np.array(all_embeddings).reshape(-1, 1)
# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(all_embeddings)

# Add cluster labels to the dataset
labels = kmeans.predict(all_embeddings)
dataset = dataset.map(lambda example, idx: {"clusters_id": int(labels[idx])},
                       with_indices=True,remove_columns=['embeddings'])

# Display the modified dataset
print(dataset)

# Iterate through each cluster ID and filter the dataset
for cluster_id in range(num_clusters):  # Assuming you have 16 clusters
    filtered_cluster = dataset.filter(lambda example: example['clusters_id'] == cluster_id)

    print(f"Cluster {cluster_id} has {len(filtered_cluster)} examples!")

    # Convert to list of dictionaries
    cluster_list = [dict(example) for example in filtered_cluster]

    # Save as JSON
    with open(f"dataset/WizardLM_evol_instruct_70k/cluster_{cluster_id}.json", 'w') as f:
        json.dump(cluster_list, f)