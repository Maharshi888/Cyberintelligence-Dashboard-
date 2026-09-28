# Practical 9: Text Mining and Web Mining
# Domain  : Cybersecurity
# Dataset : Cybersecurity_Dataset.csv (1100 threat reports)

import ast
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Folder where this script is saved (CSV and outputs live here)
BASE = Path(__file__).parent

# ==================================================
# PART 1: TEXT MINING
# ==================================================
print("=" * 50)
print("TEXT MINING")
print("=" * 50)

# Load dataset
df = pd.read_csv(BASE / "Cybersecurity_Dataset.csv")

print("\nDataset Shape:")
print(df.shape)

# ---- Word frequency on the threat description text ----
stop_words = {"the", "and", "for", "with", "from", "through", "on", "in",
              "of", "to", "a", "an", "by", "at", "is", "are"}

words = []
for description in df["Cleaned Threat Description"].astype(str):
    tokens = re.findall(r"[a-z]+", description.lower())
    words.extend([w for w in tokens if w not in stop_words and len(w) > 2])

word_freq = Counter(words)

print("\nTop 10 Words:")
for word, count in word_freq.most_common(10):
    print(f"{word} : {count}")

# ---- Threat category frequency ----
print("\nThreat Category Frequency:")
threat_freq = df["Threat Category"].value_counts()
print(threat_freq)

# ---- Extracted keyword frequency (column stores a list as text) ----
keywords = []
for item in df["Keyword Extraction"]:
    try:
        keywords.extend([k.lower() for k in ast.literal_eval(item)])
    except (ValueError, SyntaxError):
        pass

print("\nTop 5 Extracted Keywords:")
for word, count in Counter(keywords).most_common(5):
    print(f"{word} : {count}")

# ---- Attack vector and defence mechanism ----
print("\nAttack Vector Frequency:")
print(df["Attack Vector"].value_counts())

print("\nSuggested Defense Mechanism Frequency:")
print(df["Suggested Defense Mechanism"].value_counts())

# ---- Average severity per threat category ----
print("\nAverage Severity Score by Threat Category:")
print(df.groupby("Threat Category")["Severity Score"].mean().round(2))

# ---- Save text mining output ----
text_output = pd.DataFrame(word_freq.most_common(50), columns=["Word", "Frequency"])
text_output.to_csv(BASE / "text_mining_output.csv", index=False)

print("\nText mining data saved successfully.")

# ==================================================
# PART 2: WEB MINING
# ==================================================
print("\n" + "=" * 50)
print("WEB MINING")
print("=" * 50)

url = "https://en.wikipedia.org/wiki/Computer_security"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers, timeout=20)
print("\nStatus Code:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

# Page title
print("\nPage Title:")
print(soup.title.get_text(strip=True))

# Headings (h1, h2, h3)
print("\nHeadings:")
for heading in soup.find_all(["h1", "h2", "h3"]):
    text = heading.get_text(strip=True).replace("[edit]", "")
    if text:
        print("-", text)

# Extract links
print("\nExtracting Links...")
link_list = []
for a in soup.find_all("a", href=True):
    link_text = a.get_text(strip=True)
    if link_text:
        link_list.append({"Link Text": link_text, "URL": a["href"]})

links_df = pd.DataFrame(link_list)

print("\nTotal Links:")
print(len(links_df))

print("\nFirst 10 Links:")
print(links_df.head(10).to_string(index=False))

# Save web mining output
links_df.to_csv(BASE / "web_mining_output.csv", index=False)
print("\nData saved successfully as web_mining_output.csv")

print("\nPractical 9 Completed Successfully.")