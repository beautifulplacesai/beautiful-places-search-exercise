"""Generate explainer.ipynb, the live-session notebook (speaker runs it; students
can open the same notebook and follow along).

Run: uv run python make_explainer.py
The .ipynb is build output; this script is the source of truth.

Paced for a ~45 minute live session: every concept is built on screen from
the smallest possible example, with two audience rounds.
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

This is the notebook presented in the session. Open it and follow along, run
what we run. Afterwards, `tryout.ipynb` contains the same stack as exercises
you build yourself.\
""")

# ================================================================ setup
md("""\
## Setup: run this first

One cell downloads and checks everything: the photo metadata and embeddings
(in the repo), the CLIP model (~340 MB, cached after the first run), and the
local LLM for section 4 (several GB via Ollama; start this cell early, the
first three sections work even while the LLM is still downloading).

Prerequisites (see README): `uv sync` done; [Ollama](https://ollama.com/download)
installed for section 4; and for the web tool, a free
[Tavily](https://app.tavily.com) key in a `.env` file (copy `.env.example` to
`.env` and fill it in). `.env` is gitignored, so the key can never be
committed or published.\
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
    \"\"\"Local cache first, else download from Geograph (CC BY-SA, attribution in data).\"\"\"
    p = CACHE / row["photos"]
    if not p.exists():
        r = _rq.get(row["url"], timeout=20,
                    headers={"User-Agent": "beautiful-places-exercise"})
        r.raise_for_status()
        p.write_bytes(r.content)
    return Image.open(p)

def show(rows, title=None, note_col=None):
    \"\"\"Display a row of photos with name + beauty score captions.\"\"\"
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
## The data: London's verified beautiful places

≈4,800 photos of ≈4,500 verified, named places in London. Each carries a
**scenic score (0–10)** from our model, a CNN trained on **200,000+ human
beauty ratings**. This measured-beauty data is the company's core asset: no
public model or dataset provides it.\
""")

code("""\
show(photos.nlargest(6, "score"), "Highest-scoring photos in the set (scenic model)")\
""")

# ================================================================ section 1
md("""\
## 1 · From pixels to meaning

None of these photos are tagged. What we have instead: every photo has been
turned into **512 numbers** by CLIP, an open model trained on 400 million
image and caption pairs. Look at them:\
""")

code("""\
print("shape of the photo matrix:", emb.shape)
print("the first photo, first 12 of its 512 numbers:")
print(emb[0][:12].round(3))\
""")

md("""\
Meaningless to us. But treat the 512 numbers as **coordinates: a position on
a map**. Not a map of London, a map of *meanings*. CLIP arranges it so that
photos of similar things get nearby positions: swan photos end up in one
neighbourhood, churches in another.

The trick that makes search possible: CLIP can also place a **sentence** on
the same map. Watch, a sentence becomes 512 numbers of exactly the same kind:\
""")

code("""\
def embed_text(texts):
    \"\"\"One vector per sentence, in the same space as the photo vectors.\"\"\"
    with torch.no_grad():
        toks = clip.tokenize(texts, truncate=True).to(device)
        vecs = model.encode_text(toks)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs.cpu().numpy()

sentence = embed_text(["swans on a lake"])[0]
print("the sentence, first 12 of its 512 numbers:")
print(sentence[:12].round(3))\
""")

md("""\
Same shape, same map. So the question "how similar is this sentence to this
photo?" becomes measurable: **how close are their two positions?**

Try it: the sentence's position against three photos' positions. One photo
really shows swans.\
""")

code("""\
swan_photo   = emb[378]     # Wimbledon Common, swans on the water
church_photo = emb[12]      # St Pancras Old Church
street_photo = emb[307]     # Well Street Common, no swans

print("sentence vs swan photo:   ", round(float(sentence @ swan_photo), 3))
print("sentence vs church photo: ", round(float(sentence @ church_photo), 3))
print("sentence vs street photo: ", round(float(sentence @ street_photo), 3))
show(photos.iloc[[378, 12, 307]])\
""")

md("""\
The sentence's position is closest to the photo that means the same thing.
That is the whole insight. A search engine is now nothing new: measure the
sentence against **all 4,788 photos at once** and keep the closest six.
**The entire engine is five lines:**\
""")

code("""\
def search(query, k=6):
    sims = emb @ embed_text([query])[0]          # similarity to every photo
    hits = photos.iloc[np.argsort(-sims)[:k]].copy()
    hits["sim"] = np.sort(sims)[::-1][:k]
    return hits

show(search("swans on a lake"), '"swans on a lake"', note_col="sim")\
""")

md("""\
Nobody tagged these photos. The meaning is in the geometry. More:\
""")

code("""\
show(search("misty park at dawn"), '"misty park at dawn"', note_col="sim")\
""")

code("""\
show(search("autumn trees reflected in a lake"), '"autumn trees reflected in a lake"', note_col="sim")\
""")

md("""\
**Audience: suggest a query.** Descriptions of weather, light, season and
colour work particularly well.\
""")

code("""\
# audience queries here, live:
# show(search("..."), note_col="sim")\
""")

md("""\
Before moving on, one question for the room. This query worked a moment ago:
*"misty park at dawn"*. And this one works too:\
""")

code("""\
show(search("cherry blossom in spring"), '"cherry blossom in spring"', note_col="sim")\
""")

md("""\
**But CLIP has no calendar and no clock. How can it know spring, or dawn?**

It cannot. It sees blossom, soft light, long shadows, and in its 400 million
training captions those pixels co-occur with the words "spring" and "dawn".
CLIP **infers from what is visible**, and guesses at everything else. The
rule for the whole session: if it is in the pixels, CLIP can find it; if it
must be known, CLIP can only guess.

One more habit to install: the similarity numbers themselves.\
""")

code("""\
sims = emb @ embed_text(["a red brick bridge"])[0]
print(pd.Series(sims).describe().round(3))\
""")

md("""\
The best match in the whole dataset is only ~0.35, and everything is squashed
between about 0.1 and 0.35. Normal for CLIP. The numbers mean something only
**relative to each other**: rank with them, never read them as percentages.\
""")

# ================================================================ section 2
md("""\
## 2 · Where similarity search fails

Beautiful Places exists to answer one question: *where is beautiful?*
Ask the engine we just built.\
""")

code("""\
hits = search("the most beautiful park in London")
show(hits, '"the most beautiful park in London"', note_col="sim")

print(f"mean beauty of these results: {hits.score.mean():.2f}")
print(f"most beautiful photo in our data: {photos.score.max():.2f}")\
""")

md("""\
Two things went quietly wrong, and it is worth being precise:

1. **Beauty was never consulted.** These photos *resemble the words*
   "beautiful park". Their measured beauty is mediocre; the scenic scores sat
   in the metadata, unused. Similarity is not ranking.
2. **A superlative requires comparison.** "Most beautiful" means examining
   every park and ordering them, not retrieving lookalikes. And per the rule
   from section 1: "most beautiful" is not in the pixels of any single photo.

The correct answer is already in our own data. Build it in three steps,
starting with one line:\
""")

code("""\
photos.nlargest(10, "score")[["name", "score"]]\
""")

md("""\
Read the column header: this ranks **photos**. The user asked about
**places**, and a place is not one photo:\
""")

code("""\
photos.groupby("name").size().sort_values(ascending=False).head(8)\
""")

md("""\
Some places have twenty photos. A place deserves one entry with one combined
score, which raises a question that is secretly a **product decision**: is a
place's beauty its **best** photo (`max`) or its **typical** photo (`mean`)?

**Audience: shout it. Max or mean?** (While building this session, `max`
once crowned a windmill in Croydon as London's most beautiful architecture,
on the strength of one spectacular photo. Neither answer is wrong.)\
""")

code("""\
def leaderboard(subset=None, top=10):
    d = photos if subset is None else subset
    return (d.groupby("name")
             .agg(beauty=("score", "max"), photos_n=("photos", "count"),
                  lat=("latitude", "first"), lon=("longitude", "first"))
             .nlargest(top, "beauty"))

leaderboard(photos[photos.category == "nature"])\
""")

code("""\
winner = leaderboard(photos[photos.category == "nature"]).index[0]
show(photos[photos.name == winner].nlargest(3, "score"), f"Our data's answer: {winner}")\
""")

md("""\
A name, a score, evidence photos. *That* is an answer, powered by the one
thing only Beautiful Places has: measured beauty.\
""")

# ================================================================ section 3
md("""\
## 3 · Enriching the data with zero-shot classification

A limitation one level down: our data only distinguishes `nature` from
`architecture`. Ask for "the most beautiful *bridge*" and it cannot even
filter to bridges.

The embeddings fix this too. Watch the mechanism on **one photo and three
candidate labels**:\
""")

code("""\
labels = ["a photo of a bridge", "a photo of a park", "a photo of a church"]
label_vecs = embed_text(labels)

print(photos.iloc[42]["name"])
for lab, s in zip(labels, emb[42] @ label_vecs.T):
    print(f"  {s:.3f}  {lab}")
show(photos.iloc[[42]])\
""")

md("""\
The photo sits closest to one label sentence. Pick the closest and you have
classified the photo, **with no training data and no new model**. This is
called zero-shot classification. Now all 4,800 photos at once:\
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
label_vecs = embed_text(list(LABELS.values()))
photos["subtype"] = np.array(list(LABELS))[(emb @ label_vecs.T).argmax(1)]
photos["subtype"].value_counts()\
""")

code("""\
leaderboard(photos[photos.subtype == "bridge"], top=5)\
""")

md("""\
The data can now answer "most beautiful bridge / canal / cemetery": fifteen
question types it could not express a minute ago, at essentially zero cost.

**The limitation, stated plainly:** zero-shot labels are noisy. CLIP
classifies by appearance (section 1's rule again), so a round timber-framed
theatre gets labelled "pub", and a park photo with a footbridge in frame
becomes a "bridge". Good enough for a prototype and for today. **The
reliable approach, what we run in production, is a multimodal LLM that
examines every image**, or a classifier trained on labelled examples. That
costs roughly a thousand times more compute than one matrix multiply.
Choosing the right point on that cost and reliability curve is a core
engineering decision.

One gap remains: users don't query databases in pandas. They ask in English,
and a good answer includes *why* it is the answer.\
""")

# ================================================================ section 4
md("""\
## 4 · An LLM that uses our tools

We build this in steps, because each step answers a question the previous
one raises.

**Step 1: the model alone.** An open-source LLM (Gemma 4 E4B, 4.5 B
parameters, running locally on this laptop). No tools, no data. Just ask.\
""")

code("""\
from langchain_ollama import ChatOllama

llm = ChatOllama(model=MODEL, temperature=0)
print(llm.invoke("What is the most beautiful park in London? Answer in two sentences.").content)\
""")

md("""\
A confident answer, and it owes nothing to our measurements: the model is
remembering the internet. The same famous spots Google would give you. This
is exactly the problem the app exists to fix.

**Step 2: show it our data, by hand.**\
""")

code("""\
top10 = leaderboard(photos[photos.category == "nature"]).to_string()
question = f\"\"\"Here are our measured beauty scores for London nature spots:

{top10}

Based on this data, what is the most beautiful park in London? Two sentences.\"\"\"
print(llm.invoke(question).content)\
""")

md("""\
Now it is grounded in our data. Model plus proprietary data beats model
alone: the core idea behind most serious AI products.

But pasting data into every question does not scale, and we had to know in
advance which data the question would need. What if the model could fetch
data itself, when it decides it needs it?

**Step 3: tools.** A tool is a Python function with a good docstring. The
docstring is how the model decides when to call it: write it for the model.
We give it three, wrapping everything built this session, plus **Tavily**,
the same web search the production pipeline's judge uses.\
""")

code("""\
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search_photos(description: str) -> list:
    \"\"\"Find photos of London places matching a VISUAL description (looks, light,
    season). Returns place names, beauty scores and similarity.\"\"\"
    hits = search(description, k=6)
    return hits[["name", "score", "sim"]].round(3).to_dict("records")

@tool
def beauty_db(subtype: str = None, top: int = 10) -> list:
    \"\"\"Rank named London places by MEASURED beauty (our scenic model, 0-10).
    Use for any 'most beautiful / prettiest / best' question.
    subtype (optional) filters by place type, one of: church, bridge, palace,
    castle, pub, canal, riverside, park, lake, woodland, street, monument,
    windmill, cemetery, modern.\"\"\"
    d = photos if subtype is None else photos[photos.subtype == subtype]
    return leaderboard(d, top).reset_index().round(2).to_dict("records")

@tool
def web_search(query: str) -> list:
    \"\"\"Live web search: opening times, entry fees, what a place is known for.\"\"\"
    r = _rq.post("https://api.tavily.com/search",
                 json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3})
    return [{"title": h["title"], "content": h["content"][:300]}
            for h in r.json().get("results", [])]\
""")

md("""\
**Step 4: the agent.** The model reads the question and *decides which tools
to call*. There is no routing if-statement anywhere in this code. The system
prompt sets the rules of engagement, including the one that took us longest
to learn: the database's answer is final, even when it is not famous. Hidden
gems are the product.\
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

agent = create_agent(llm, [search_photos, beauty_db, web_search], system_prompt=SYSTEM)

def ask(question):
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
Compare with Step 1: same question, same model, and now a measured answer
with a reason. Watch the trace line: the model chose the tool by itself.

**One system, several question types.** No search code changes between the
next three questions; the model routes:\
""")

code("""\
ask("What is the most beautiful bridge in London?")\
""")

md("""\
Not a landmark: a small footbridge in a Morden park. Measured beauty
surfacing what postcards miss is precisely the product.\
""")

code("""\
ask("What's the most beautiful garden in London, and is it free to get in?")\
""")

md("""\
Read the trace: it **chained**. First the database (which garden), then the
web (is it free). Different tools for different halves of one question.

**Audience: ask the system a question.** Anything a person looking for a
beautiful place would ask. Watch which tools it selects, and judge whether
the stated reasoning follows from the tool results.\
""")

code("""\
# audience questions here, live:
# ask("...")\
""")

# ================================================================ section 5
md("""\
## 5 · Failure cases worth remembering

Real outputs from building this notebook. Nothing crashes when these happen:
the system simply returns something wrong, with confidence. This is why
production systems verify, and why evaluation matters more than demos.\
""")

code("""\
# Zero-shot classification error: the highest-scoring "pub" in London,
# according to CLIP, is Shakespeare's Globe, a round, timber-framed theatre.
# Classification by appearance alone confuses visually similar categories.
show(photos[photos.subtype == "pub"].nlargest(3, "score"), "CLIP's highest-scoring 'pubs'")\
""")

code("""\
# Ranking design error: with max aggregation, the answer to "most beautiful
# architecture in London" was once Shirley Windmill: one exceptional photo
# outranked every photo of every famous building. Whether a place is ranked
# by its best photo or its average is a product decision with visible
# consequences.
show(photos[photos.subtype == "windmill"].nlargest(3, "score"), "The windmill that outranked the cathedrals")\
""")

md("""\
What connects them: pixels carry *appearance*, not *facts*, and aggregate
statistics encode *design choices*, not truths.

### How this is actually solved when it matters

When you genuinely need structured data attached to images or places, not
for a demo but to publish, the standard tooling has shifted generation by
generation:

- **Supervised classifiers (the standard until ~2022).** One model per
  attribute, trained on thousands of hand-labelled examples. Our scenic
  model is this generation: a CNN trained on 200,000+ human beauty ratings.
  Reliable within a fixed taxonomy; but every new attribute means a new
  labelling effort and a new model.
- **Zero-shot embeddings (section 3).** Instant and taxonomy-free, but it
  classifies by appearance. In practice this tier is used for prototyping,
  search, deduplication and *pre-filtering*, not for facts you would
  publish.
- **Multimodal LLMs under careful guidance (the current standard, and what
  our pipeline runs).** A vision LLM examines each image and must return a
  structured JSON record against a schema, with domain rules encoded in the
  prompt. Its confidence routes every result: high-confidence accepted
  automatically, uncertain cases escalated to web research, the rest to
  human review. The "careful guidance" is the engineering: output schemas,
  calibrated thresholds, and code-side enforcement. A prompt is a wish; the
  checks live in code.
- **Distillation closes the loop.** Once the LLM has labelled enough data
  well, you train a small, cheap classifier on its outputs, recovering
  tier-one economics at scale, with tier-three quality supervision.

Each tier costs orders of magnitude more per image than the one before, so
real systems are funnels: cheap methods handle the volume so that expensive
models, and humans, only ever see the hard cases.

Your notebook, `tryout.ipynb`, contains this exact stack as exercises: you
will build the leaderboard yourselves, design your own classification
labels, add a geographic tool, and make the agent explain itself better than
this one does. The TA has hints. We would genuinely like to see your best
results, and your most instructive failures.\
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "explainer.ipynb")
print(f"Wrote explainer.ipynb with {len(cells)} cells")
