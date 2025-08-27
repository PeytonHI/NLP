import torch
from sentence_transformers import util

def is_similar_query(bilstm_encoder, claim, content):
    claim_tensor = torch.tensor(claim, dtype=torch.float32).unsqueeze(0)
    content_tensor = torch.tensor(content, dtype=torch.float32)

    claim_encoding = bilstm_encoder(claim_tensor)
    content_encoding = bilstm_encoder(content_tensor)

    similarity = util.pytorch_cos_sim(claim_encoding, content_encoding)
    return similarity.item()

# Other helper functions like search_wikipedia, getTopArticles, etc.
