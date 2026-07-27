"""Generate tryout.ipynb, the student exercise notebook.

Run: uv run python make_tryout.py
The .ipynb is build output; this script is the source of truth.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------- intro
md("""\
# Find Me a Beautiful Place: your turn

You watched this system get built. Now it's yours. The notebook is a sequence
of exercises: each part stands on its own, difficulty is marked with ⭐, and it's
fine not to finish; choose where to spend your time. The TA has hints (and
solutions, but ask for a hint first: that's where the learning is).

| part | what you build | difficulty |
|---|---|---|
| 0 | run the search engine | (just run it) |
| 1 | probe the embedding space | ⭐ |
| 2 | the beauty leaderboard | ⭐⭐ |
| 3 | teach it geography ("near me") | ⭐⭐ |
| 4 | invent your own labels (zero-shot CLIP) | ⭐⭐ |
| 5 | give an AI your tools (LangChain) | ⭐⭐ |
| 6 | make it explain WHY | ⭐⭐⭐ |
| 7 | live web knowledge | ⭐⭐⭐ |
| 8 | advanced extensions (pick one) | ⭐⭐⭐⭐ |

**The data:** ≈4,800 photos of ≈4,500 verified beautiful places in London,
each with a scenic score (0–10) from Beautiful Places' model, a CNN trained on
200,000+ human beauty ratings. Photos © Geograph contributors (CC BY-SA),
loaded on demand from geograph.org.uk.

**Before you start** (see README for details): `uv sync` done, and for Parts 4+
Ollama installed with a model pulled (`ollama pull gemma4:e4b`, or `gemma4:e2b`
on a low-RAM laptop, then change `MODEL` below).\
""")

# ---------------------------------------------------------------- part 0
md("""\
## Part 0: Run the search engine  *(run these cells, nothing to write)*

The first cell loads the data and defines two helpers: `search(query)` for CLIP
semantic search over every photo, and `show(rows)` to display results (photos
download from Geograph on first view, then cache in `data/img_cache/`).\
""")

code("""\
import io, requests as _rq
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch, clip

DATA = Path("data")
CACHE = DATA / "img_cache"; CACHE.mkdir(exist_ok=True)

photos = pd.read_parquet(DATA / "london5k_index.parquet")
emb = np.load(DATA / "london5k_embeddings.npy")

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
model, _ = clip.load("ViT-B/32", device=device)   # same model Beautiful Places uses
MODEL = "gemma4:e4b"                              # LLM for parts 4+ ("gemma4:e2b" if low RAM)

def fetch_image(row):
    \"\"\"Local cache first, else download from Geograph (CC BY-SA).\"\"\"
    p = CACHE / row["photos"]
    if not p.exists():
        r = _rq.get(row["url"], timeout=20,
                    headers={"User-Agent": "beautiful-places-exercise"})
        r.raise_for_status()
        p.write_bytes(r.content)
    return Image.open(p)

def show(rows, title=None, note_col=None):
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
    if title: fig.suptitle(title, fontsize=13, y=1.04)
    plt.tight_layout(); plt.show()

def search(query, k=6):
    with torch.no_grad():
        vec = model.encode_text(clip.tokenize([query], truncate=True).to(device))
        vec = (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    sims = emb @ vec
    hits = photos.iloc[np.argsort(-sims)[:k]].copy()
    hits["sim"] = np.sort(sims)[::-1][:k]
    return hits

show(search("golden autumn light on a quiet path"), note_col="sim")
print(f"{len(photos):,} photos · {photos.name.nunique():,} places. You just searched all of them.")\
""")

# ---------------------------------------------------------------- part 1
md("""\
## Part 1 ⭐: Probe the space

The search box is yours. Find:

1. a query that works remarkably well (weather, light, season, colour…)
2. a query where the results **surprise** you
3. a query that **fails**, then form a theory: *why* did it fail? Is the answer
   something you can SEE in a photo, or something you'd have to KNOW?

The `sim` number is cosine similarity. On this data, winners live around
0.28–0.34. **Mini-challenge:** what's the highest similarity you can provoke?
What kind of query maxes it out, and what does that tell you about how CLIP
was trained?\
""")

code("""\
show(search("your query here"), note_col="sim")\
""")

code("""\
# more experiments...
\
""")

# ---------------------------------------------------------------- part 2
md("""\
## Part 2 ⭐⭐: The beauty leaderboard

In the demo you saw the crack: ask for "the most beautiful park" and similarity
search returns *lookalikes*, ignoring the measured beauty sitting in the data.

Your job: build the function that answers superlatives properly.

**Write `leaderboard(category, top)`** returning the top places ranked by beauty:
group `photos` by `name`, aggregate the score (one place can have several photos.
Decide: `max` or `mean`? Argue your choice), count photos, sort, return the top.

Columns you'll want: `name`, `score`, `category`, `latitude`, `longitude`.\
""")

code("""\
def leaderboard(category=None, top=10):
    d = photos if category is None else photos[photos.category == category]
    # TODO: group by place name, aggregate beauty + photo count + lat/lon,
    #       return the `top` most beautiful as a DataFrame
    ...

leaderboard("nature")\
""")

code("""\
# ✅ check yourself: this should print a named place with beauty > 6.5
best = leaderboard("nature").index[0]
print("Most beautiful nature spot:", best)
show(photos[photos.name == best].nlargest(3, "score"), best)

# now: what's the most beautiful ARCHITECTURE in London?
# and: does your max-vs-mean choice change the winner? try both.\
""")

# ---------------------------------------------------------------- part 3
md("""\
## Part 3 ⭐⭐: Teach it geography

"Beautiful" is only half a user's question. The other half is usually "**near
me**". You're at King's College London, Strand campus: `51.5115, -0.1160`.

**Write `places_near(lat, lon, km)`** returning photos within `km` of a point,
sorted by beauty. The maths you need (haversine distance) is started for you.

**Challenge:** what's the most beautiful place within 2 km of campus, somewhere
you could actually walk to after this session? Check its photo. Do you believe
the model?\
""")

code("""\
def haversine_km(lat1, lon1, lat2, lon2):
    \"\"\"Distance in km between two points (vectorised over arrays).\"\"\"
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

KCL = (51.5115, -0.1160)

def places_near(lat, lon, km=2.0, top=6):
    # TODO: compute each photo's distance to (lat, lon) with haversine_km,
    #       keep rows within `km`, sort by beauty, return the top
    ...

show(places_near(*KCL, km=2), "Most beautiful within 2 km of KCL")\
""")

# ---------------------------------------------------------------- part 4
md("""\
## Part 4 ⭐⭐: Invent your own labels

The dataset only knows two categories: `nature` and `architecture`. But in the
demo you saw the trick: **the embeddings you already have give you classification
for free.** Write labels as sentences, embed them, and each photo's nearest label
becomes its type: zero-shot classification, one matrix multiply.

The demo used concrete labels (church, bridge, canal…). Yours don't have to be
concrete. Subjective labels like `romantic` or `eerie` are a genuinely
interesting experiment: does CLIP encode mood, or only appearance?

**Do:** design your own label set (5–10), classify all photos, then answer a
question the raw data never could. For instance, the most beautiful *eerie*
place in London.

**Honesty checkpoint (from the demo):** zero-shot labels are noisy: CLIP judges
by *looks* (it thinks Shakespeare's Globe is a pub). The reliable-but-expensive
fix is a vision LLM looking at every image, which is what Beautiful Places runs
in production. Know which rung of that ladder you're standing on.\
""")

code("""\
MY_LABELS = {
    # "label": "a photo of ...",   <- write yours!
    "romantic": "a photo of a romantic place for a date",
    "eerie":    "a photo of an eerie, atmospheric place",
}
with torch.no_grad():
    lab = model.encode_text(clip.tokenize(list(MY_LABELS.values())).to(device))
    lab = (lab / lab.norm(dim=-1, keepdim=True)).cpu().numpy()

photos["subtype"] = np.array(list(MY_LABELS))[(emb @ lab.T).argmax(1)]
print(photos["subtype"].value_counts())

# most beautiful place per YOUR label:
# TODO: reuse your Part 2 leaderboard on photos[photos.subtype == "..."]\
""")

# ---------------------------------------------------------------- part 5
md("""\
## Part 5 ⭐⭐: Give an AI your tools

Now the demo's finale, but with **your** functions inside. We use LangChain, the
framework Beautiful Places' production pipeline runs on. A **tool** is just a
Python function with a good docstring: the docstring is how the model decides
when to call it, so write it for the *model*, not for humans.

Needs Ollama running (`ollama serve`, usually automatic) with your `MODEL` pulled.

Wire up the agent below (the `@tool` wrappers are started for you; finish the
TODOs by delegating to your Part 2/3 functions), then ask it the big question.\
""")

code("""\
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

@tool
def search_photos(description: str) -> list:
    \"\"\"Find photos of London places matching a VISUAL description (looks, light,
    season). Returns place names, beauty scores and similarity.\"\"\"
    hits = search(description, k=6)
    return hits[["name", "score", "sim"]].round(3).to_dict("records")

@tool
def beauty_leaderboard(category: str = "nature", top: int = 10) -> list:
    \"\"\"Rank named London places by MEASURED beauty (scenic model, 0-10).
    Use for any 'most beautiful / prettiest / best' question.
    category: 'nature' or 'architecture'.\"\"\"
    # TODO: return your Part 2 leaderboard as a list of dicts
    #       (hint: .reset_index().round(2).to_dict("records"))
    ...

@tool
def near_me(lat: float, lon: float, km: float = 2.0) -> list:
    \"\"\"Most beautiful verified places within `km` of a coordinate, best first.\"\"\"
    # TODO: return your Part 3 places_near() as a list of dicts
    ...

llm = ChatOllama(model=MODEL, temperature=0)
agent = create_agent(llm, [search_photos, beauty_leaderboard, near_me],
                     system_prompt="You answer questions about beautiful places in "
                                   "London using the tools. Never guess.")

def ask(question, a=None):
    for step in (a or agent).stream({"messages": [("user", question)]}, stream_mode="values"):
        m = step["messages"][-1]
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  🔧 {tc['name']}({tc['args']})")
        elif m.type == "ai" and m.content:
            print(f"\\n{m.content}")

ask("What is the most beautiful park in London?")\
""")

md("""\
**Watch the trace.** Did it pick the right tool? Now try:
`ask("I'm at KCL on the Strand. Where's somewhere beautiful I can walk to?")`
does it call `near_me` with sensible coordinates? (Where did it get them from?
That's the model's world knowledge filling an argument, spooky, and worth
discussing with your TA.)\
""")

# ---------------------------------------------------------------- part 6
md("""\
## Part 6 ⭐⭐⭐: Make it explain WHY

An answer without a reason is just a lookalike: the failure from the demo
wearing a suit. Rewrite the agent's **system prompt** so that every answer contains:

1. the place name **and its measured beauty score**,
2. **WHY**, reasoning grounded in what the tools returned (not vibes),
3. a practical tip (best time, how to get there, what to look for).

Then ask the *same question* before and after. Which answer would you ship in the
app? What single sentence in your prompt made the biggest difference? (Prompt
engineering is an experimental science: change one thing at a time.)

A real failure to defend against: while building this exercise, the agent once
received the database's honest answer, *distrusted it because it wasn't famous*,
replaced it with a well-known bridge found via web search, and invented a
score for it. Write a rule that prevents exactly this, then try to provoke it.\
""")

code("""\
MY_SYSTEM = \"\"\"You are the Beautiful Places guide for London.
# TODO: your rules here. Require the score. Require the WHY. Require the tip.
# Tell it when to use which tool. Forbid guessing.
\"\"\"

agent2 = create_agent(llm, [search_photos, beauty_leaderboard, near_me],
                      system_prompt=MY_SYSTEM)

ask("What is the most beautiful park in London?", agent2)\
""")

# ---------------------------------------------------------------- part 7
md("""\
## Part 7 ⭐⭐⭐: Live knowledge

Your agent knows beauty (our scores) and looks (CLIP). It doesn't know *today*:
opening hours, entry fees, events. Add a **web search tool**: Tavily, the same
search tool you saw inside the production pipeline in the lecture.

**Get your own free API key (2 minutes):**
1. Go to **[app.tavily.com](https://app.tavily.com)** and sign up (free, no card
   needed; the free tier gives 1,000 searches/month, plenty for today).
2. Copy the API key from your dashboard (starts with `tvly-`).
3. Copy the file `.env.example` to `.env` and put your key in it. `.env` is
   in `.gitignore`, so the key stays on your machine and can never be
   committed or shared by accident.

Then ask something no offline dataset can answer:

> "What's the most beautiful garden in London, and is it free to get in?"

Watch the trace: the good agent chains `beauty_leaderboard` → `web_search`. If
yours doesn't, whose fault is it: the model's, or your docstrings'?\
""")

code("""\
import os
from dotenv import load_dotenv
load_dotenv()                        # reads the gitignored .env file
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
assert TAVILY_KEY, "Copy .env.example to .env and add your key from app.tavily.com"

@tool
def web_search(query: str) -> list:
    \"\"\"Live web search, opening times, entry fees, what a place is known for.\"\"\"
    r = _rq.post("https://api.tavily.com/search",
                 json={"api_key": TAVILY_KEY, "query": query, "max_results": 3})
    return [{"title": h["title"], "content": h["content"][:300]}
            for h in r.json().get("results", [])]

agent3 = create_agent(llm, [search_photos, beauty_leaderboard, near_me, web_search],
                      system_prompt=MY_SYSTEM)

ask("What's the most beautiful garden in London, and is it free to get in?", agent3)\
""")

# ---------------------------------------------------------------- part 8
md("""\
## Part 8 ⭐⭐⭐⭐: Advanced extensions (pick one)

**A. The duplicate problem.** Search "riverside sunset": near-identical photos of
the same spot crowd the top. Implement **MMR** (maximal marginal relevance):
iteratively pick results that are similar to the *query* but dissimilar to
*already-picked results* (`score = λ·sim_to_query − (1−λ)·max_sim_to_picked`,
all from `emb`). One extra function, big product win.

**B. The model that can see.** Gemma 4 is *multimodal*. Send it your top-4 photos
and the user's query, and let it pick the single best match, with a reason.
You've built a two-stage ranker: cheap CLIP recall → smart visual rerank.
(Hint: the `ollama` python package, `ollama.chat(model=MODEL, messages=[...],
images=[path])`, the cached files in `data/img_cache/` are ready to send.)

**C. The ranking formula.** "Most beautiful park" ignores relevance; "misty park"
ignores beauty. Build `hybrid_search(query, alpha)` ranking by
`alpha·similarity + (1−alpha)·normalised_beauty`. Plot how the top-6 changes as
alpha goes 0→1. Which alpha would you ship? Why?

**Whichever you pick**, also inspect your Part 4 classification's failure
cases, misclassifications reveal what CLIP actually encodes (the demo's
example: Shakespeare's Globe classified as a pub).

Keep your best result and your most instructive failure. We'd like to see both.\
""")

code("""\
# boss level workspace
\
""")

md("""\
## What you just built

Cheap vector recall → proprietary ranking data → an LLM that orchestrates tools
and explains itself. That's not a toy: it's the architecture of essentially every
serious AI product being shipped right now, and it's how search will work in
Beautiful Places. If your agent gave an answer you'd actually ship, we genuinely
want to see it.\
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "tryout.ipynb")
print(f"Wrote tryout.ipynb with {len(cells)} cells")
