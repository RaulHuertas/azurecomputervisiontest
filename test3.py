from sentence_transformers import SentenceTransformer

# Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model = SentenceTransformer("all-mpnet-base-v2")
#embeddings = model.encode([
#    "The weather is lovely today.",
#    "It's so sunny outside!",
#    "He drove to the stadium.",
#])

embeddings = model.encode([
    #references
    "BOL NUMBER",
    "INVOICE",
    "FACTURA",
    "BILL OF LADING",
    #scanned data
    #"BOL NUMBER",
    "Invoice Number",
    "BOL",
    "nothing",
    "non related stuff",
])

similarities = model.similarity(embeddings, embeddings)

print(similarities)

