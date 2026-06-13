from pymilvus import MilvusClient

client = MilvusClient(uri="http://127.0.0.1:19530")

collection = "equipment_faq_hybrid_v1"
query = "设备点检需要注意什么？"

print("collections =", client.list_collections())
print("describe =", client.describe_collection(collection))

print("\n=== sparse text only search ===")
print(client.search(
    collection_name=collection,
    data=[query],
    anns_field="sparse",
    search_params={"metric_type": "BM25", "params": {}},
    limit=5,
    output_fields=["text"],
))