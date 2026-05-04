import requests

URL = "https://api.openalex.org/works"

params = {
    "search": "deep learning finance time series forecasting",
    "filter": "from_publication_date:2020-01-01",
    "per_page": 200  # fetch more to sort locally
}

response = requests.get(URL, params=params)
data = response.json()

results = data.get("results", [])

papers = []

for paper in results:
    papers.append({
        "title": paper.get("title"),
        "year": paper.get("publication_year"),
        "citations": paper.get("cited_by_count", 0),
        "doi": paper.get("doi")
    })

# Sort by citations descending
papers = sorted(papers, key=lambda x: x["citations"], reverse=True)

# Take top 20
top_papers = papers[:20]

# Print results
for i, p in enumerate(top_papers, 1):
    print(f"{i}. {p['title']} ({p['year']})")
    print(f"   Citations: {p['citations']}")
    print(f"   DOI: {p['doi']}")
    print("-" * 80)
