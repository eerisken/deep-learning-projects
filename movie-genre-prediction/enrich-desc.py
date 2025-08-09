from openai import OpenAI
import json

client = OpenAI()

def enrich_description(desc, genres):
    prompt = (
        f"Expand this movie description to better reflect its genres: {', '.join(genres)}.\n"
        f"Keep it under 80 words. Avoid adding unrelated details.\n\n"
        f"Description: {desc}\nExpanded:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

# Example
print(enrich_description("A young boy discovers a hidden portal in his backyard.", ["Fantasy", "Adventure"]))
