from embeddings import get_embedding_client

client = get_embedding_client()
embedding = client.create_embedding("§ 160.101 Statutory basis and purpose.\n\nThe requirements of this subchapter implement sections 1171-1180 of the Social Security Act (the Act), sections 262 and 264 of Public Law 104-191, section 105 of Public Law 110-233, sections 13400-13424 of Public Law 111-5, and section 1104 of Public Law 111-148.")

print(embedding)