"""Generate explainer.ipynb, the live-session notebook (speaker runs it; students
can open the same notebook and follow along).

Run: uv run python make_explainer.py
The .ipynb is build output; this script is the source of truth.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------- title
md("""\
# Find Me a Beautiful Place
### Building search for Beautiful Places, live, in one notebook

This is the notebook presented in the session. Open it and follow along, run
what we run. Afterwards, `tryout.ipynb` contains the same stack as exercises
you build yourself.

> **Speaker notes (session ≈ 45 min):** The lecture covered what Beautiful
> Places is. This session builds the product's missing feature, search, from
> first principles, and examines where each technique works and where it fails.
> Two audience-participation points are marked.\
""")

# ---------------------------------------------------------------- setup
md("""\
## Setup: run this first

One cell downloads and checks everything: the photo metadata and embeddings
(in the repo), the CLIP model (~340 MB, cached after the first run), and the
local LLM for section 4 (several GB via Ollama; start this cell early; sections
1–3 work even while the LLM is still downloading).

Prerequisites (see README): `uv sync` done; [Ollama](https://ollama.com/download)
installed for section 4; a free [Tavily](https://app.tavily.com) key in the
`TAVILY_API_KEY` environment variable if you want the web tool.\
""")

code("""\
import io, os, subprocess
import requests as _rq
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch, clip

DATA = Path("data")
CACHE = DATA / "img_cache"; CACHE.mkdir(exist_ok=True)

photos = pd.read_parquet(DATA / "london5k_index.parquet")
emb = np.load(DATA / "london5k_embeddings.npy")     # one 512-dim vector per photo

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
model, _ = clip.load("ViT-B/32", device=device)     # same model our pipeline uses

MODEL = "gemma4:e4b"            # section 4 LLM; use "gemma4:e2b" on a low-RAM laptop
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")   # free key: app.tavily.com

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

# ---------------------------------------------------------------- data
md("""\
## The data: London's verified beautiful places

≈5,000 photos of ≈4,700 verified, named places in London. Every photo is
attached to a named place. Some were verified automatically on strong evidence,
the rest by a human. Each carries a **scenic score (0–10)** from our model, a
CNN trained on **200,000+ human beauty ratings**. This measured-beauty data is
the company's core asset: no public model or dataset provides it.

> **Speaker notes:** Make the provenance concrete in one line (Geograph photos,
> CC-licensed, GPS-tagged, verified through our pipeline) and stress that the
> scores come from a trained vision model, not an LLM's opinion.\
""")

code("""\
show(photos.nlargest(6, "score"), "Highest-scoring photos in the set (scenic model)")\
""")

# ---------------------------------------------------------------- part 1
md("""\
## 1 · Semantic search with a joint embedding space

None of these photos are tagged. CLIP, an open-source model trained on 400
million image–caption pairs, maps **images and sentences into the same
512-dimensional vector space**, arranged so that things which *mean* the same
end up close together. Our photos are already in that space (the `emb` matrix,
precomputed). Searching is then just geometry: embed the query sentence, and
rank every photo by cosine similarity.

> **Speaker notes:** One mental model to draw: a cloud of points (the photos); a
> query becomes a new point; search is nearest-neighbour lookup. No keywords, no
> tags, no index: the meaning is in the geometry.\
""")

code("""\
def search(query, k=6):
    with torch.no_grad():
        vec = model.encode_text(clip.tokenize([query], truncate=True).to(device))
        vec = (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    sims = emb @ vec                      # cosine similarity to every photo
    hits = photos.iloc[np.argsort(-sims)[:k]].copy()
    hits["sim"] = np.sort(sims)[::-1][:k]
    return hits

show(search("misty park at dawn"), '"misty park at dawn"', note_col="sim")\
""")

code("""\
show(search("autumn trees reflected in a lake"), '"autumn trees reflected in a lake"', note_col="sim")
show(search("cherry blossom in spring"), '"cherry blossom in spring"', note_col="sim")\
""")

md("""\
**Audience: suggest a query.** Descriptions of weather, light, season, and
colour work particularly well: the model is matching *visual* content.

> **Speaker notes:** Take two or three suggestions and type them live. This is
> where confidence in embeddings should peak. Then the pivot: "So is search
> solved? Let's ask the question this company exists to answer."\
""")

code("""\
# audience queries here, live:
# show(search("..."), note_col="sim")\
""")

# ---------------------------------------------------------------- part 2
md("""\
## 2 · Where similarity search fails

Beautiful Places exists to answer one question: *where is beautiful?*
Ask the search engine directly.\
""")

code("""\
hits = search("What is the most beautiful park in London?")
show(hits, '"What is the most beautiful park in London?"', note_col="sim")

print(f"mean beauty of these results: {hits.score.mean():.2f}")
print(f"most beautiful photo in our data: {photos.score.max():.2f}")\
""")

md("""\
Two distinct failures, and it's worth being precise about them:

1. **Similarity is not ranking.** These photos *resemble the words* "beautiful
   park", but their measured beauty is mediocre (compare the mean above with
   the dataset maximum). The scenic scores sat in the metadata, unused, because
   cosine similarity has no notion of them.
2. **A superlative requires comparison.** "Most beautiful" means examining every
   park and ordering them, not retrieving lookalikes. More generally, properties
   like "hidden", "peaceful", or "quiet" are facts about places, not patterns of
   pixels; no visual embedding can rank by them.

> **Speaker notes:** This is the central technical point of the session:
> *similarity ≠ ranking; retrieval ≠ answering.* Give it a beat.

The correct answer is already in our own data. It takes five lines:\
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

# ---------------------------------------------------------------- part 3: enrich
md("""\
## 3 · Enriching the data with zero-shot classification

There's a limitation one level down: our database only distinguishes `nature`
from `architecture`. Ask for "the most beautiful *bridge*" and it cannot even
filter to bridges.

The embeddings we computed for search solve this too. Write category labels as
sentences, embed them, and assign each photo its nearest label. This is
**zero-shot classification**: a working 15-way image classifier from one matrix
multiplication, with no training data and no new model.\
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
with torch.no_grad():
    lab = model.encode_text(clip.tokenize(list(LABELS.values())).to(device))
    lab = (lab / lab.norm(dim=-1, keepdim=True)).cpu().numpy()

photos["subtype"] = np.array(list(LABELS))[(emb @ lab.T).argmax(1)]
photos["subtype"].value_counts()\
""")

code("""\
leaderboard(photos[photos.subtype == "bridge"], top=5)\
""")

md("""\
The database can now answer "most beautiful bridge / canal / cemetery"
fifteen question types it couldn't express a minute ago, at essentially zero
cost.

**The limitation, stated plainly:** zero-shot labels are noisy. CLIP classifies
by appearance, so a round timber-framed theatre gets labelled "pub", and a park
photo with a footbridge in frame becomes a "bridge". This accuracy level is
acceptable for a prototype and for today's exercise. **The reliable approach
what we run in production, is a multimodal LLM that examines every image**
(our pipeline uses a Gemma vision model to caption and verify each photo), or a
classifier trained on labelled examples. That costs roughly a thousand times
more compute than one matrix multiply. Choosing the right point on that
cost–reliability curve is a core engineering decision, not a detail.

> **Speaker notes:** This is the honest-limitations moment for classification
> present the trade-off (instant but noisy vs. expensive but reliable) as the
> lesson itself. Concrete failure examples are shown at the end of the session.

One gap remains: users don't query databases in pandas. They ask in English, and
a good answer includes *why* it's the answer.\
""")

# ---------------------------------------------------------------- part 4
md("""\
## 4 · An LLM that uses our tools

We give an open-source LLM (Gemma 4 E4B, 4.5 B parameters, running locally on
this laptop) three **tools**: Python functions it may choose to call. The
framework is **LangChain**, the same one our production pipeline uses, and the
web tool is **Tavily**, the same search tool the pipeline's judge uses.

| tool | wraps | contributes |
|---|---|---|
| `search_photos` | section 1's embedding search | visual matching |
| `beauty_db` | section 2's leaderboard + section 3's subtypes | measured beauty |
| `web_search` | Tavily web API | current facts |

The model reads the question and *decides which tools to call*. There is no
routing if-statement anywhere in this code, that decision, and the explanation
the model gives, is what the LLM contributes.

> **Speaker notes:** This is the tool-use / agent pattern that underlies most
> current AI products, shown in three cells. Point at the tool-call trace as it
> streams. Note that the system prompt *requires* a WHY, an answer without
> grounding is exactly the failure from section 2 in different clothes.\
""")

code("""\
from langchain.tools import tool

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
    \"\"\"Live web search, opening times, entry fees, what a place is known for.\"\"\"
    r = _rq.post("https://api.tavily.com/search",
                 json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3})
    return [{"title": h["title"], "content": h["content"][:300]}
            for h in r.json().get("results", [])]\
""")

code("""\
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

SYSTEM = \"\"\"You are the Beautiful Places guide for London. Use the tools, never guess.
For any 'most beautiful / prettiest / best' question, consult beauty_db (with the
right subtype). The database's answer is final: it reflects measured beauty, and
unfamiliar or small places outranking famous ones is expected and correct, these
hidden gems are the product. Never substitute a famous place from your own
knowledge or from the web, and never state a score that did not come verbatim
from beauty_db. Use search_photos for visual descriptions. Use web_search ONLY
for practical facts (opening times, entry fees) about a place you have already
selected from the database.

Always answer with:
1. The place name and its measured beauty score (from beauty_db, verbatim).
2. WHY, one or two sentences of reasoning grounded in the tool results.
3. A practical tip (how to visit, best time, what to look for).\"\"\"

llm = ChatOllama(model=MODEL, temperature=0)
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
### One system, several question types

The beauty score is no longer a column in a CSV, it's a tool the model reaches
for; the embeddings enriched the database so that tool understands *kinds* of
places. The same system now handles question types that pure similarity search,
or a pure LLM, would each fail at alone:

> **Speaker notes:** Run both. Name what routes where as each streams:
> measurement with the new subtypes; then measurement plus live web facts. Worth
> saying explicitly: no search code changed between these questions, the model
> selects the strategy. On the bridge answer: it's a small footbridge in Morden
> Hall Park, not a landmark, measured beauty surfacing what postcards miss is
> precisely the product.\
""")

code("""\
ask("What is the most beautiful bridge in London?")\
""")

code("""\
ask("What's the most beautiful garden in London, and is it free to get in?")\
""")

md("""\
**Audience: ask the system a question.** Anything a person looking for a
beautiful place would ask. Watch which tools it selects, and judge whether its
stated reasoning actually follows from the tool results.\
""")

code("""\
# audience questions here, live:
# ask("...")\
""")

# ---------------------------------------------------------------- failure cases
md("""\
## 5 · Failure cases worth remembering

> **Speaker notes (~3 min):** Close on honest limitations, these are real
> outputs from building this notebook. The underlying point: these errors are
> *silent*. Nothing crashes; the system simply returns something wrong with
> full confidence. That is why production systems verify with stronger models
> and human review, and why evaluation matters more than demos.\
""")

code("""\
# Zero-shot classification error: the highest-scoring "pub" in London,
# according to CLIP, is Shakespeare's Globe, a round, timber-framed theatre.
# Classification by appearance alone confuses visually similar categories.
show(photos[photos.subtype == "pub"].nlargest(3, "score"), "CLIP's highest-scoring 'pubs'")\
""")

code("""\
# Ranking design error: before subtypes, the database's answer to "most
# beautiful architecture in London" was Shirley Windmill, one exceptional
# photo outranked every photo of every famous building. Whether a place is
# ranked by its best photo or its average is a product decision with visible
# consequences.
show(photos[photos.subtype == "windmill"].nlargest(3, "score"), "The windmill that outranked the cathedrals")\
""")

md("""\
What connects them: pixels carry *appearance*, not *facts*, and aggregate
statistics encode *design choices*, not truths.

### How this is actually solved when it matters

When you genuinely need structured data attached to images or places, not for a
demo, but to publish, the standard tooling has shifted generation by generation:

- **Supervised classifiers (the standard until ~2022).** One model per
  attribute, trained on thousands of hand-labelled examples. Our scenic model is
  this generation: a CNN trained on 200,000+ human beauty ratings.
  Reliable within a fixed taxonomy; but every new attribute means a new
  labelling effort and a new model.
- **Zero-shot embeddings (section 3).** Instant and taxonomy-free, but it
  classifies by appearance. In practice this tier is used for prototyping,
  search, deduplication and *pre-filtering*, not for facts you would publish.
- **Multimodal LLMs under careful guidance (the current standard, and what our
  pipeline runs).** A vision LLM examines each image and must return a
  structured JSON record against a schema, caption, place type, viewing
  distance, with domain rules encoded in the prompt. Its confidence routes
  every result: high-confidence accepted automatically, uncertain cases
  escalated to web research, the rest to human review. The "careful guidance"
  is the engineering: output schemas, calibrated thresholds, and code-side
  enforcement, a prompt is a wish; the checks live in code.
- **Distillation closes the loop.** Once the LLM has labelled enough data well,
  you train a small, cheap classifier on its outputs, recovering tier-one
  economics at scale, with tier-three quality supervision.

Each tier costs orders of magnitude more per image than the one before, so real
systems are funnels: cheap methods handle the volume so that expensive models
and humans, only ever see the hard cases.

> **Speaker notes:** This list is the session's "what would I actually use"
> takeaway, worth slowing down for. It is also literally our pipeline: Gemma
> vision captioning into JSON, confidence-thresholded escalation, web research,
> human review before anything is published.

> **Speaker notes, handover (last 2 min):** "Your notebook, `tryout.ipynb`,
> contains this exact stack as a sequence of exercises: you'll build the
> leaderboard yourselves, design your own classification labels, add a
> geographic tool, and improve the agent's reasoning. The TA has hints and can
> discuss any part in depth. I'd genuinely like to see your best results, and
> your most instructive failures."\
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "explainer.ipynb")
print(f"Wrote explainer.ipynb with {len(cells)} cells")
