"""Generate explainer.ipynb, the live-session notebook (speaker runs it; students
can open the same notebook and follow along).

Run: uv run python make_explainer.py
The .ipynb is build output; this script is the source of truth.

Paced for a ~45 minute live session. Written for an international audience:
short sentences, no jargon, every concept built from the smallest example.
Code cells read line by line; no clever one-liners.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ================================================================ title
md("""\
# Find Me a Beautiful Place
### Building search for Beautiful Places, live, in one notebook

## 👉 github.com/beautifulplacesai/beautiful-places-search-exercise

To follow along on your own laptop: open that page and **follow the README
instructions**. Setup takes a few minutes; start now and let it run.

This is the notebook presented in the session. Run what we run. Afterwards,
`tryout.ipynb` in the same folder has today's ideas as exercises you build
yourself.\
""")

# ================================================================ setup
md("""\
## Setup: run this first

This one cell loads everything: the photo data, the CLIP model (~340 MB,
downloads once), and the local LLM used in section 4 (several GB via Ollama;
start early, sections 1 to 3 work while it downloads).

Prerequisites (see README): `uv sync` done; [Ollama](https://ollama.com/download)
installed for section 4; and for the web tool, a free
[Tavily](https://app.tavily.com) key in a `.env` file (copy `.env.example` to
`.env`, then edit `.env` in your code editor; dot-files are hidden in the
file browser). `.env` is gitignored, so the key cannot be published by
accident.\
""")

code("""\
import os, subprocess
import requests as _rq
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import torch, clip

load_dotenv()                                       # reads the gitignored .env file
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

DATA = Path("data")
CACHE = DATA / "img_cache"; CACHE.mkdir(exist_ok=True)

photos = pd.read_parquet(DATA / "london5k_index.parquet")
emb = np.load(DATA / "london5k_embeddings.npy")

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
model, _ = clip.load("ViT-B/32", device=device)     # same model our pipeline uses

MODEL = "gemma4:e4b"            # section 4 LLM; use "gemma4:e2b" on a low-RAM laptop

def llm_ready():
    \"\"\"Check Ollama is running and MODEL is present; pull it if missing.\"\"\"
    try:
        tags = _rq.get("http://localhost:11434/api/tags", timeout=3).json()
        if not any(m["name"].startswith(MODEL) for m in tags.get("models", [])):
            print(f"Downloading {MODEL} via Ollama (several GB, a few minutes)...")
            subprocess.run(["ollama", "pull", MODEL], check=True)
        return True
    except Exception as e:
        print(f"Ollama not reachable ({type(e).__name__}). Sections 1-3 work fine; "
              f"install and start Ollama before section 4.")
        return False

def fetch_image(row):
    \"\"\"Local cache first, else download from Geograph (CC BY-SA, credits in data).\"\"\"
    p = CACHE / row["photos"]
    if not p.exists():
        r = _rq.get(row["url"], timeout=20,
                    headers={"User-Agent": "beautiful-places-exercise"})
        r.raise_for_status()
        p.write_bytes(r.content)
    return Image.open(p)

def show(rows, title=None, note_col=None):
    \"\"\"Display photos with name + beauty score captions.\"\"\"
    rows = rows.head(6)
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.4))
    axes = [axes] if len(rows) == 1 else axes
    for ax, (_, r) in zip(axes, rows.iterrows()):
        try:
            ax.imshow(fetch_image(r))
        except Exception:
            ax.text(0.5, 0.5, "image\\nunavailable", ha="center", va="center")
        cap = f'{str(r["name"])[:30]}\\nbeauty {r["score"]:.1f}'
        if note_col is not None:
            cap += f' · {r[note_col]:.3f}'
        ax.set_title(cap, fontsize=9); ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13, y=1.04)
    plt.tight_layout(); plt.show()

print(f"{len(photos):,} photos · {photos.name.nunique():,} named places · "
      f"scores {photos.score.min():.1f}-{photos.score.max():.1f} · CLIP on {device} · "
      f"LLM {'ready' if llm_ready() else 'pending'} · "
      f"Tavily {'configured' if TAVILY_API_KEY else 'not set'}")\
""")

# ================================================================ data
md("""\
## The data

4,788 photos of beautiful places in London. Each photo has:

- a **place name** (St James's Park Lake, Tower Bridge, ...)
- a location (latitude, longitude)
- a **beauty score from 0 to 10**, given by our own model. That model is a
  CNN trained on 200,000+ ratings made by real people. Nobody else has this
  data. It is the company's core asset.\
""")

code("""\
show(photos.nlargest(6, "score"), "The highest-scoring photos in the data")\
""")

# ================================================================ section 1
md("""\
## 1 · How can a computer search photos?

The photos have no tags and no descriptions. So how can we search them?

The answer: every photo has already been converted into a list of
**512 numbers**. The conversion was done by CLIP, an open model trained on
400 million pairs of (image, caption). Here is what those numbers look like:\
""")

code("""\
print("we have", emb.shape[0], "photos, each converted to", emb.shape[1], "numbers")
print()
print("the first photo's numbers (first 10 of 512):")
print(emb[0][:10].round(3))\
""")

md("""\
The numbers mean nothing to us. But they follow one rule:

> **Photos that show similar things get similar numbers.**

Think of each photo's numbers as its **position on a map**. Not a map of
London. A map of *content*: all lake photos sit close together in one region,
all skyscrapers sit together in another region, far away.

Here is the trick that makes search possible. CLIP can also convert a
**sentence** into 512 numbers, a position on the **same map**:\
""")

code("""\
def embed_text(texts):
    \"\"\"Convert sentences to positions on the same map as the photos.\"\"\"
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        vecs = model.encode_text(toks)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs.cpu().numpy()

sentence = embed_text(["swans on a lake"])[0]
print("the sentence 'swans on a lake' (first 10 of its 512 numbers):")
print(sentence[:10].round(3))\
""")

md("""\
The sentence "swans on a lake" is now a position on the photo map. Where did
it land? **Near the photos that show swans on lakes.** That is what CLIP was
trained to do.

Let's check. We measure how close the sentence is to three very different
photos: a lake with swans, a glass skyscraper, and a quiet street.
The number is closeness: **bigger means closer**.

Before running: which photo should get the biggest number?\
""")

code("""\
swan_lake  = emb[378]    # photo of Wimbledon Common: swans on the water
skyscraper = emb[1247]   # photo of the Gherkin tower, City of London
street     = emb[3281]   # photo of Ensor Mews: a quiet cobbled street

print("closeness of each photo to the sentence 'swans on a lake':")
print("   swan lake photo: ", round(float(swan_lake @ sentence), 3))
print("   skyscraper photo:", round(float(skyscraper @ sentence), 3))
print("   street photo:    ", round(float(street @ sentence), 3))

show(photos.iloc[[378, 1247, 3281]])\
""")

md("""\
The swan lake wins by a lot (0.32 against 0.11 and 0.14). The sentence landed
where it should: next to the photo that shows the same thing.

And that is all a search engine is. Measure the sentence's closeness to
**all 4,788 photos**, and show the six closest:\
""")

code("""\
def search(query, k=6):
    sims = emb @ embed_text([query])[0]                # closeness to every photo
    hits = photos.iloc[np.argsort(-sims)[:k]].copy()   # the k closest photos
    hits["sim"] = np.sort(sims)[::-1][:k]
    return hits

show(search("swans on a lake"), '"swans on a lake"', note_col="sim")\
""")

md("""\
Nobody tagged these photos. The search works through position on the map
alone. More examples:\
""")

code("""\
show(search("misty park at dawn"), '"misty park at dawn"', note_col="sim")\
""")

code("""\
show(search("autumn trees reflected in a lake"), '"autumn trees reflected in a lake"', note_col="sim")\
""")

md("""\
**Audience: suggest a query.** Weather, light, season and colour all work
well.\
""")

code("""\
# audience queries here, live:
# show(search("..."), note_col="sim")\
""")

md("""\
One more query, and then a question:\
""")

code("""\
show(search("cherry blossom in spring"), '"cherry blossom in spring"', note_col="sim")\
""")

md("""\
It found spring. **But CLIP has no calendar. How can it know the season?**

It cannot. It sees blossom and soft light in the pixels. In its 400 million
training captions, those things appear together with the word "spring". So
the word "spring" and blossom photos sit close together on the map. CLIP is
**guessing the season from what is visible**, and the guess is usually good.

Remember this rule for the whole session:

> **If it is visible in the pixels, CLIP can find it. If it must be known,
> CLIP can only guess.**

Last detail: what do the closeness numbers mean? Look at all 4,788 of them
for one query:\
""")

code("""\
sims = emb @ embed_text(["a red brick bridge"])[0]
print("smallest:", sims.min().round(3))
print("average: ", sims.mean().round(3))
print("largest: ", sims.max().round(3))\
""")

md("""\
Even the best match is only about 0.35. That is normal for CLIP. The numbers
are not percentages and are not scores out of 1. They only mean something
**compared to each other**. Use them to rank, nothing else.\
""")

# ================================================================ section 2
md("""\
## 2 · Where search fails

Beautiful Places exists to answer one question: *where is beautiful?*
Ask our new search engine:\
""")

code("""\
hits = search("the most beautiful park in London")
show(hits, '"the most beautiful park in London"', note_col="sim")

print("average beauty score of these results:", round(hits.score.mean(), 2))
print("highest beauty score in our data:     ", round(photos.score.max(), 2))\
""")

md("""\
Nice photos. But look at the two numbers: these results average about **4.9**,
while our data contains places scoring up to **6.9**. The search engine never
looked at the beauty scores at all. It cannot. Closeness-on-the-map only
finds photos that *look like the words* "beautiful park".

And remember section 1's rule: "most beautiful" is not visible in the pixels
of any single photo. To answer it, you must **compare all the parks**, using
the scores.

So let's answer it properly, with the scores. First try, one line:\
""")

code("""\
photos.nlargest(10, "score")[["name", "score"]]\
""")

md("""\
This is a top-10 of **photos**. But the user asked for the best **place**,
and a place is not one photo. Some places appear in our data many times:\
""")

code("""\
photos_per_place = photos.groupby("name").size()
photos_per_place.sort_values(ascending=False).head(8)\
""")

md("""\
Up to 20 photos of one place. So: to rank places, first collect each place's
photos together, then give the place **one combined score**. We take the
place's **best photo** as its score (`max` in the code). Keep that choice in
mind; we will question it in a moment.\
""")

code("""\
def leaderboard(subset=None, top=10):
    d = photos if subset is None else subset
    grouped = d.groupby("name").agg(
        beauty=("score", "max"),        # a place scores as its BEST photo
        photos_n=("photos", "count"),   # how many photos it has
        lat=("latitude", "first"),
        lon=("longitude", "first"),
    )
    return grouped.nlargest(top, "beauty")

def show_best_photo_of_each(places, title=None):
    \"\"\"For each place in a leaderboard, show its best photo.\"\"\"
    rows = (photos[photos.name.isin(places.index)]
            .sort_values("score", ascending=False)
            .drop_duplicates("name"))
    show(rows, title)

leaderboard(photos[photos.category == "nature"])\
""")

code("""\
top5_nature = leaderboard(photos[photos.category == "nature"], top=5)
show_best_photo_of_each(top5_nature, "The top 5 nature places, best photo of each")\
""")

md("""\
A name, a score, photos as evidence. This is a real answer, and only our
data can give it.

One detail worth noticing: **all five winners are water**. Nobody told the
model to prefer water. It learned that from 200,000+ human ratings, and the
research behind our scenic model found the same: water and trees drive
scenicness. The data is showing you what people find beautiful.

Now the promised question about that `max`. Run the same leaderboard for
architecture:\
""")

code("""\
leaderboard(photos[photos.category == "architecture"], top=5)\
""")

md("""\
Numbers are hard to judge. Look at the five places themselves, best photo
of each:\
""")

code("""\
top5_architecture = leaderboard(photos[photos.category == "architecture"], top=5)
show_best_photo_of_each(top5_architecture, "The top 5 architecture places, best photo of each")\
""")

md("""\
**Number one is a windmill in Croydon.** One spectacular photo put it above
every famous building in London.

Is that right or wrong? With `max`, one great photo can crown a place. With
`mean` (the average), places with many normal photos get pulled down. Neither
is mathematically wrong. It is a **product decision**, and it changes what
users see.

**Audience: keep the windmill at number one, or demote it?**\
""")

md("""\
### Beautiful, and close by

"Most beautiful" is one question users ask. The other one is: **what is
beautiful near me?** Our data has every photo's location, so this is one
filter away. The distance formula for two points on the Earth is called
haversine; take it as given.

One honest simplification: "me" is **hardcoded to this room**. We typed the
coordinates of King's College into the notebook. In the real app, your
phone's GPS supplies them instead; everything else works exactly the same.\
""")

code("""\
def haversine_km(lat1, lon1, lat2, lon2):
    \"\"\"Distance in km between two points on Earth. Works on whole columns.\"\"\"
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

HERE = (51.5115, -0.1160)      # King's College London, Strand campus

photos["km_away"] = haversine_km(HERE[0], HERE[1],
                                 photos["latitude"], photos["longitude"])

walkable = photos[photos["km_away"] <= 2.0]
show_best_photo_of_each(leaderboard(walkable, top=5),
                        "The most beautiful places within 2 km of this room")\
""")

md("""\
Beauty scores plus one distance filter: a walking recommendation. Try one
after the session.\
""")

# ================================================================ section 3
md("""\
## 3 · Creating new categories from the map

Our data has only two categories: `nature` and `architecture`. If a user
asks for the most beautiful **bridge**, we are stuck: nothing in the data
says which photos show bridges.

But we can create that information ourselves, with the map from section 1.
The idea, on one photo:

1. Write each possible category as a sentence: "a photo of a bridge",
   "a photo of a park", "a photo of a church".
2. Put those three sentences on the map, like we did with the search query.
3. Check which sentence the photo sits **closest** to. That is its category.\
""")

code("""\
bridge_label = embed_text(["a photo of a bridge"])[0]
park_label   = embed_text(["a photo of a park"])[0]
church_label = embed_text(["a photo of a church"])[0]

one_photo = emb[42]                     # a photo we have not looked at yet

print("closeness of this photo to each label:")
print("   'a photo of a bridge':", round(float(one_photo @ bridge_label), 3))
print("   'a photo of a park':  ", round(float(one_photo @ park_label), 3))
print("   'a photo of a church':", round(float(one_photo @ church_label), 3))

show(photos.iloc[[42]])\
""")

md("""\
The photo is closest to "a photo of a park", and it is indeed a park
(Priory Gardens). We just gave a photo a category **without training
anything**.

Now the same for all 4,788 photos, with 15 categories. Three steps in the
code: put all 15 label sentences on the map, measure every photo against
every label, and give each photo the label it sits closest to.\
""")

code("""\
LABELS = {
    "church":    "a photo of a church or cathedral",
    "bridge":    "a photo of a bridge",
    "palace":    "a photo of a palace or grand stately home",
    "castle":    "a photo of a castle or fortress",
    "pub":       "a photo of a traditional pub",
    "canal":     "a photo of a canal or dock with boats",
    "riverside": "a photo of a wide river with a city skyline",
    "park":      "a photo of a park or garden with trees and lawns",
    "lake":      "a photo of a lake or pond",
    "woodland":  "a photo of a forest or woodland path",
    "street":    "a photo of a pretty street with houses",
    "monument":  "a photo of a monument, statue or memorial",
    "windmill":  "a photo of a windmill",
    "cemetery":  "a photo of an old cemetery with gravestones",
    "modern":    "a photo of modern glass architecture or skyscrapers",
}

label_positions = embed_text(list(LABELS.values()))   # 15 positions on the map
closeness = emb @ label_positions.T                   # every photo vs every label
winner = closeness.argmax(axis=1)                     # per photo: closest label

photos["subtype"] = np.array(list(LABELS))[winner]
photos["subtype"].value_counts()\
""")

md("""\
Every photo now has one of 15 categories. And the leaderboard from section 2
immediately becomes more useful. A question our data could not even express
a minute ago: the most beautiful canal in London?\
""")

code("""\
leaderboard(photos[photos.subtype == "canal"], top=5)\
""")

code("""\
top5_canals = leaderboard(photos[photos.subtype == "canal"], top=5)
show_best_photo_of_each(top5_canals, "The top 5 canals, best photo of each")\
""")

md("""\
Look at the photos: five out of five really are canals, narrowboats
included. Regent's Canal wins.

This technique is called **zero-shot classification**: classifying with no
training examples, only label sentences. It cost us one matrix multiply.
It is not always this accurate: CLIP judges by looks (section 1's rule),
and in section 5 you will count its mistakes yourself. Good enough for a
prototype. In production we do this job with a stronger, more expensive
model that actually examines each image.

One thing is still missing from our system. Users do not write pandas. They
ask questions in their own words, in their own language, and they deserve an
answer with a **reason**.\
""")

# ================================================================ section 4
md("""\
## 4 · An LLM that uses our tools

We build this in small steps.

**Step 1: the LLM alone.** An open model (Gemma 4), running on this laptop.
No tools, no access to our data. Just the question:\
""")

code("""\
from langchain_ollama import ChatOllama

llm = ChatOllama(model=MODEL, temperature=0)
answer = llm.invoke("What is the most beautiful park in London? Answer in two sentences.")
print(answer.content)\
""")

md("""\
A confident answer, naming famous parks. The model is remembering what the
internet says, exactly like Google. Our measurements played no part. This is
the "same famous spots" problem our app exists to fix.

**Step 2: give it our data, by hand.** Paste the leaderboard into the
question:\
""")

code("""\
top10 = leaderboard(photos[photos.category == "nature"]).to_string()

question = f\"\"\"Here are our measured beauty scores for London nature spots:

{top10}

Based on this data only: what is the most beautiful park in London? Two sentences.\"\"\"

answer = llm.invoke(question)
print(answer.content)\
""")

md("""\
Now the answer uses **our data**. This is the big idea: *a model plus data it
never had*. But pasting data by hand does not scale. We would need to know,
for every question, which data to paste.

Better: let the model **fetch data itself**, when it decides it needs it.

**Step 3: build the three tools the model will need.**

A tool is a normal Python function that the model is allowed to call. We
will build one tool for each ability from today, plus one new one:

| tool | what it does | from |
|---|---|---|
| `search_photos` | find photos that match a description | section 1 |
| `beauty_db` | rank places by measured beauty, by category | sections 2 and 3 |
| `near_me` | the most beautiful places within walking distance | section 2 |
| `web_search` | look up live facts on the web (Tavily, same as our pipeline) | new |

One important detail before the code. Each function starts with a short
description in quotes. That text is not decoration: **it is how the model
decides which tool to use for which question.** Read the four descriptions
below with that in mind.

Each data tool also displays the photos of what it returns, so you can see
what the model sees before it answers.\
""")

code("""\
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search_photos(description: str) -> list:
    \"\"\"Find photos of London places matching a VISUAL description (looks, light,
    season). Returns place names, beauty scores and similarity.\"\"\"
    hits = search(description, k=6)
    show(hits, note_col="sim")            # display what the model sees
    return hits[["name", "score", "sim"]].round(3).to_dict("records")

@tool
def beauty_db(subtype: str = None, top: int = 10) -> list:
    \"\"\"Rank named London places by MEASURED beauty (our scenic model, 0-10).
    Use for any 'most beautiful / prettiest / best' question.
    subtype (optional) filters by place type, one of: church, bridge, palace,
    castle, pub, canal, riverside, park, lake, woodland, street, monument,
    windmill, cemetery, modern.\"\"\"
    d = photos if subtype is None else photos[photos.subtype == subtype]
    top_places = leaderboard(d, top)
    show_best_photo_of_each(top_places)   # display what the model sees
    return top_places.reset_index().round(2).to_dict("records")

@tool
def near_me(km: float = 2.0) -> list:
    \"\"\"The most beautiful places within km of the configured user location,
    best first. Use when the user asks what is nearby or within walking
    distance.\"\"\"
    walkable = photos[photos["km_away"] <= km]
    top_places = leaderboard(walkable, top=5)
    show_best_photo_of_each(top_places)   # display what the model sees
    return top_places.reset_index().round(2).to_dict("records")

@tool
def web_search(query: str) -> list:
    \"\"\"Live web search: opening times, entry fees, what a place is known for.\"\"\"
    r = _rq.post("https://api.tavily.com/search",
                 json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3})
    return [{"title": h["title"], "content": h["content"][:300]}
            for h in r.json().get("results", [])]\
""")

md("""\
One design note on `near_me`: it is hardcoded to King's College. Run this
notebook at home in Tokyo and it still answers as if you were on the Strand.
In the real app, the phone's GPS replaces our typed constant; nothing else
changes. In both cases the location comes from the application, never from
the model. The model's only decisions are whether to use the tool, and the
radius.\
""")

md("""\
**Step 4: the agent.** Connect the model and the tools. The model reads each
question and **chooses which tool to call**. We wrote no rules for choosing.

The instructions below (the "system prompt") set its behaviour. One rule took
us longest to learn, so read it in full: *the database's answer is final,
even when it is not famous*. Without it, the model sometimes ignored our data
and answered "Tower Bridge" from memory, with an invented score.\
""")

code("""\
SYSTEM = \"\"\"You are the Beautiful Places guide for London. Use the tools; never guess.
For any 'most beautiful / prettiest / best' question, consult beauty_db (with the
right subtype). The database's answer is final: it reflects measured beauty, and
unfamiliar or small places outranking famous ones is expected and correct. These
hidden gems are the product. Never substitute a famous place from your own
knowledge or from the web, and never state a score that did not come verbatim
from beauty_db. Use search_photos for visual descriptions. Use web_search ONLY
for practical facts (opening times, entry fees) about a place you have already
selected from the database.

Always answer with:
1. The place name and its measured beauty score (from beauty_db, verbatim).
2. WHY: one or two sentences of reasoning grounded in the tool results.
3. A practical tip (how to visit, best time, what to look for).\"\"\"

agent = create_agent(llm, [search_photos, beauty_db, near_me, web_search], system_prompt=SYSTEM)

def ask(question):
    \"\"\"Send a question to the agent. Print which tools it calls, then its answer.\"\"\"
    for step in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
        m = step["messages"][-1]
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  tool call: {tc['name']}({tc['args']})")
        elif m.type == "ai" and m.content:
            print(f"\\n{m.content}")

ask("What is the most beautiful park in London?")\
""")

md("""\
Compare with Step 1: the same model, the same question. Now it calls our
database by itself and answers with a measured score and a reason.

Two more questions. Notice: we change no code between them. The model
**routes** each question to the right tools.\
""")

code("""\
ask("What is the most beautiful bridge in London?")\
""")

md("""\
Not a famous landmark: a small footbridge in a Morden park. Finding beautiful
places that fame forgot is exactly what the product is for.\
""")

code("""\
ask("I want a short walk. What is beautiful near me?")\
""")

code("""\
ask("What's the most beautiful garden in London, and is it free to get in?")\
""")

md("""\
Look at the tool calls above: the model **chained** two tools. First our
database (which garden is most beautiful), then the web (is it free). Two
different halves of one question, answered from two different sources.

**Audience: ask the system a question.** Anything a person looking for a
beautiful place would ask. Watch which tools it picks.\
""")

code("""\
# audience questions here, live:
# ask("...")\
""")

# ================================================================ section 5
md("""\
## 5 · When it goes wrong

Everything above worked. Now two real failures from building this notebook.
Nothing crashed in either case. The system simply gave a wrong answer, with
full confidence. This is why real products verify their outputs, and why
testing matters more than demos.

**Failure 1.** Section 3's categories come from looks alone. The canals
looked perfect. Now ask for the top "bridge" places:\
""")

code("""\
top5_bridges = leaderboard(photos[photos.subtype == "bridge"], top=5)
show_best_photo_of_each(top5_bridges, "The top 5 'bridge' places, best photo of each")\
""")

md("""\
**Count the actual bridges.** Gallows Bridge, yes. Morden Hall Park has a
small white footbridge. But the Pergola is a garden walkway, and two of the
five are simply pretty rivers. CLIP saw water, arches and railings, and
guessed "bridge".

Its funniest single mistake: the highest-scoring "pub" in London, according
to CLIP:\
""")

code("""\
show(photos[photos.subtype == "pub"].nlargest(3, "score"), "CLIP's top 'pubs'")\
""")

md("""\
Number one is **Shakespeare's Globe**: a theatre. It is round, timber-framed
and old-looking, so on the map of looks it sits near the pubs. Categories by
appearance fail exactly where appearance misleads.

**Failure 2.** The agent itself. Ask it a leading question, one that
invites it to ignore our data:\
""")

code("""\
ask("Surely the most beautiful bridge in London is Tower Bridge? Check properly.")\
""")

md("""\
Today it holds: it checks the database and stays with its answer.

But when we first built this agent, it did not hold. Asked about bridges, it
received the database's honest answer, decided a small park footbridge could
not be right, replaced it with the famous Tower Bridge from its memory, and
**invented a beauty score for it**. Nothing crashed. The answer simply
looked right and was wrong.

One sentence in the system prompt fixed it. Scroll up and read it again:
*"The database's answer is final."*

What both failures share: they are **silent**. A wrong category, a wrong
answer, delivered with full confidence.

### How this is done for real

When correct labels really matter, this is the ladder companies climb. Each
step costs roughly a thousand times more per image than the one before:

1. **Trained classifiers.** One model per category, trained on thousands of
   hand-labelled examples. Reliable, but every new category needs new
   labelled data. Our beauty model is this kind.
2. **Zero-shot labels** (what we did in section 3). Instant and free, but
   judges by looks. Used for prototypes, search and pre-filtering.
3. **A large vision model that examines every image** (what our production
   pipeline does). It must answer in a fixed format, its confidence is
   checked, uncertain cases go to web research, and the rest go to a human.
4. **Humans** review what the machines are unsure about.

Cheap methods handle the millions; expensive methods and people see only the
hard cases. You saw this same funnel in the lecture.

Your notebook, `tryout.ipynb`, has everything from today as exercises: you
will build the leaderboard yourself, invent your own categories, give the
agent a geography tool, and write a better system prompt than ours. The TA
has hints. We would love to see your best answers, and your best failures.

## 👉 github.com/beautifulplacesai/beautiful-places-search-exercise

Everything (this notebook, the exercises, the data) is in that repository.
Follow the README instructions there, then open `tryout.ipynb`.\
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "explainer.ipynb")
print(f"Wrote explainer.ipynb with {len(cells)} cells")
