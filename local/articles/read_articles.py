import requests
import csv

URL = "https://api.openalex.org/works"

params = {
    "search": "deep learning finance time series forecasting",
    "per_page": 30
}

response = requests.get(URL, params=params)
data = response.json()

results = data.get("results", [])

papers = []

for paper in results:
    title = paper.get("title")
    year = paper.get("publication_year")
    citations = paper.get("cited_by_count")
    doi = paper.get("doi")

    papers.append({
        "title": title,
        "year": year,
        "citations": citations,
        "doi": doi
    })

# Print results
for p in papers:
    print(f"{p['title']} ({p['year']}) - citations: {p['citations']}")
    print(f"DOI: {p['doi']}")
    print("-" * 80)

# Save to CSV (optional)
with open("papers.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "year", "citations", "doi"])
    writer.writeheader()
    writer.writerows(papers)

print("Saved to papers.csv")
