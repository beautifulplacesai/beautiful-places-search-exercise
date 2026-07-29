"""Generate tryout.ipynb: the student exercise notebook.

Run: uv run python make_tryout.py
The .ipynb is build output; this script is the source of truth.

Design: one idea per cell. Every concept starts with the smallest possible
example, then grows one piece at a time. Short explanation before each code
cell, observation after it.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ================================================================ intro
md("""\
# Find Me a Beautiful Place: your turn

You watched this system get built. Now you build it yourself, piece by piece.

The notebook is a sequence of parts, from guided to open-ended. Difficulty is
marked with ⭐. It is fine not to finish; later parts stand on their own, so
choose where to spend your time. The TA has hints (and solutions, but ask for
a hint first: that's where the learning is).

| part | what you build | difficulty |
|---|---|---|
| 0 | run the search engine | (just run it) |
| 1 | find out what CLIP actually sees | ⭐ |
| 2 | from photos to places: the leaderboard | ⭐⭐ |
| 3 | teach it geography | ⭐⭐ |
| 4 | invent your own categories | ⭐⭐ |
| 5 | an AI that uses your tools | ⭐⭐⭐ |
| 6 | make it explain WHY | ⭐⭐⭐ |
| 7 | live web knowledge | ⭐⭐⭐ |
| 8 | extensions | ⭐⭐⭐⭐ |

**The data:** ≈4,800 photos of ≈4,500 verified beautiful places in London,
each with a scenic score (0–10) from Beautiful Places' model, a CNN trained on
over 1.5 million human beauty ratings of 217,000 photos. Photos © Geograph contributors (CC BY-SA),
loaded on demand from geograph.org.uk.

**Before you start** (see README for details): `uv sync` done, and for Part 5
onwards, Ollama installed with the model pulled (`ollama pull granite4:3b`,
a 2 GB download that runs on any laptop).\
""")

# ================================================================ part 0
md("""\
## Part 0: Run the search engine  *(nothing to write, but do read)*

The next cell loads everything: the photo list, and one **512-number vector
per photo**, precomputed with CLIP. Those vectors are the whole trick, and
the rest of the morning is about what you can do with them.\
""")

code("""\
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import requests as _rq
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import torch, open_clip

load_dotenv()                          # reads the gitignored .env file (Part 7)
DATA = Path("data")
CACHE = DATA / "img_cache"; CACHE.mkdir(exist_ok=True)

photos = pd.read_parquet(DATA / "london5k_index.parquet")
emb = np.load(DATA / "london5k_embeddings.npy")

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
model, _, _ = open_clip.create_model_and_transforms(   # the model Beautiful Places uses
    "ViT-B-32-quickgelu", pretrained="openai", device=device)
tokenize = open_clip.get_tokenizer("ViT-B-32-quickgelu")
MODEL = "granite4:3b"                             # LLM for Part 5+: 2 GB, runs on any laptop

def fetch_image(row):
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

print(f"{len(photos):,} photos loaded · embeddings shape {emb.shape} · CLIP on {device}")\
""")

md("""\
Now the search engine itself. **Read it before you run it.** It really is
this small:

1. turn the query sentence into a vector, using the same model that embedded
   the photos
2. measure how close that vector is to every photo vector (one matrix
   multiply)
3. return the closest photos\
""")

code("""\
def embed_text(texts):
    \"\"\"One vector per sentence, in the same space as the photo vectors.\"\"\"
    with torch.no_grad():
        toks = tokenize(texts).to(device)
        vecs = model.encode_text(toks)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs.cpu().numpy()

def search(query, k=6):
    sims = emb @ embed_text([query])[0]          # similarity to every photo
    hits = photos.iloc[np.argsort(-sims)[:k]].copy()
    hits["sim"] = np.sort(sims)[::-1][:k]
    return hits

show(search("swans on a lake"), note_col="sim")\
""")

md("""\
Nobody tagged these photos with "swan". The match comes from meaning, not
keywords.

**Do:** change the query above to anything you like and run it again. Then
move on.\
""")

# ================================================================ part 1
md("""\
## Part 1 ⭐: What does CLIP actually see?

In the demo we pictured CLIP's numbers as positions on a **map of content**:
photos and sentences land on the same map, and near means "means the same".
But CLIP only ever sees **pixels**. So what exactly can a query
mention and still work? Let's find its limits, from easy to strange.

First, **things you can point at**:\
""")

code("""\
show(search("a red brick bridge"), note_col="sim")\
""")

code("""\
# your turn: two more concrete queries (objects, animals, buildings...)
\
""")

md("""\
Next, **looks that aren't objects**: light, weather, the mood of the sky.\
""")

code("""\
show(search("golden evening light"), note_col="sim")
show(search("fog over water"), note_col="sim")\
""")

md("""\
Still works. Now the strange one. Before you run the next cell, think:
**CLIP has no calendar. Can it know what season a photo was taken in?**\
""")

code("""\
show(search("cherry blossom in spring"), note_col="sim")\
""")

md("""\
It works, but not because CLIP knows the date. It sees **blossom and soft
light**, and in its 400 million training captions those pixels co-occur with
the word "spring". CLIP **infers the season from what is visible**. It is a
guess, and usually a good one.

This is the demo's rule, worth repeating: **if it is visible in the pixels,
CLIP can find it. If it must be known, CLIP can only guess.**

**Do:** probe the boundary. Try queries like these, and your own:

- "a garden in the morning"  vs  "a garden in the evening"
- "a park on a Tuesday"
- "a peaceful place with nobody around"

Find one query that works better than you expected, and one that fails even
though it sounds visual. For each, write one sentence below: what was in the
pixels, and what had to be guessed?\
""")

code("""\
# probe here
show(search("a garden in the evening"), note_col="sim")\
""")

md("""\
*(double-click to edit and write your two sentences here)*

1. Worked surprisingly well: ...
2. Failed although it sounded visual: ...\
""")

md("""\
One last thing for this part: the similarity numbers. Are they scores out of
1.0? Run the cell:\
""")

code("""\
sims = emb @ embed_text(["a red brick bridge"])[0]
print(pd.Series(sims).describe().round(3))\
""")

md("""\
Even the best match is only about 0.3, and the whole dataset is squashed
between roughly 0.1 and 0.35. That is normal for CLIP. The lesson: the
numbers only mean something **relative to each other**. Rank with them;
never treat them as percentages.\
""")

# ================================================================ part 2
md("""\
## Part 2 ⭐⭐: From photos to places

Time to break the CLIP search engine. Ask it the question this company exists to
answer:\
""")

code("""\
hits = search("the most beautiful park in London")
show(hits, note_col="sim")
print(f"mean beauty of these results: {hits.score.mean():.2f}")
print(f"best beauty score in the data: {photos.score.max():.2f}")\
""")

md("""\
Pleasant photos, mediocre scores. The search found photos that **look like
the words** "beautiful park". It never once consulted the beauty scores
sitting in the data. Similarity is not ranking.

The real answer is in the metadata. Start with one line:\
""")

code("""\
photos.nlargest(10, "score")[["name", "score"]]\
""")

md("""\
That looks like an answer, but read the column header: it ranks **photos**.
The user asked about **places**, and a place is not one photo. How many
photos can a single place have here?

**Do:** find out. One line: group `photos` by `"name"`, then `.size()`, then
`.sort_values(ascending=False)`. Show the top 10.\
""")

code("""\
# your line here
\
""")

md("""\
Some places have up to twenty photos. So a place deserves **one entry** with
**one combined score**, and that raises a question that is secretly a
**product decision**: is a place's beauty its **best** photo (`max`) or its
**typical** photo (`mean`)? You saw this live in the demo: `max` crowned a
windmill in Croydon as London's most beautiful built-up place, on the strength
of one spectacular photo. Neither choice is wrong. You choose, and you own
the consequences.

**Do:** write `leaderboard(category=None, top=10)`:

1. filter to the category if one is given (`photos.category` is `nature` or
   `built-up`)
2. group by `name`; aggregate the score (your choice of max or mean), a photo
   count, and `latitude`/`longitude` (take the first of each)
3. return the `top` places by beauty\
""")

code("""\
def leaderboard(category=None, top=10):
    d = photos if category is None else photos[photos.category == category]
    # your groupby here
    ...

leaderboard("nature")\
""")

code("""\
# ✅ check: a named place with beauty above 6.5 should win
best = leaderboard("nature").index[0]
print("Most beautiful nature spot:", best)
show(photos[photos.name == best].nlargest(3, "score"))

# then try: leaderboard("built-up")
# and: does switching max/mean change the winner?\
""")

# ================================================================ part 3
md("""\
## Part 3 ⭐⭐: Teach it geography

"Beautiful" is only half of what people ask. The other half is "**near me**".
In this notebook "me" means the coordinates we type below; in a real app the
phone's GPS would supply them. The maths for distance on a sphere is
called the haversine formula; here it is, ready-made:\
""")

code("""\
def haversine_km(lat1, lon1, lat2, lon2):
    \"\"\"Distance in km between two points. Works on whole columns at once.\"\"\"
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))

HERE = (51.5115, -0.1160)         # central London, on the Strand
ST_PAULS = (51.5138, -0.0984)     # St Paul's Cathedral

# sanity check: this should print roughly 1.2 km
print(haversine_km(HERE[0], HERE[1], ST_PAULS[0], ST_PAULS[1]).round(2), "km")\
""")

md("""\
**Do (one line):** give every photo a distance from that spot. Create a column
`photos["km_away"]` using `haversine_km` with `HERE` and the
`photos.latitude` / `photos.longitude` columns. It works on whole columns at
once: no loop needed.\
""")

code("""\
# your line here

photos.nsmallest(5, "km_away")[["name", "km_away", "score"]]\
""")

md("""\
**Do:** now combine distance and beauty. Write `places_near(km=2.0, top=6)`
returning the most beautiful photos within `km` of campus, best first. Then
answer: where should this class walk after the session?\
""")

code("""\
def places_near(km=2.0, top=6):
    ...

show(places_near(km=2.0))\
""")

md("""\
Shrink the radius to 0.5 km and run it again. Fewer, worse options: that is
the trade-off every location product lives with, radius against quality.
Where would you set it for our app?\
""")

# ================================================================ part 4
md("""\
## Part 4 ⭐⭐: Invent your own categories

The data has only two categories: `nature` and `built-up`. Ask it for
"the most beautiful *bridge*" and it cannot even filter to bridges.

The vectors fix this too. Watch the mechanism on **one photo and three
candidate labels**:\
""")

code("""\
labels = ["a photo of a bridge", "a photo of a park", "a photo of a church"]
label_vecs = embed_text(labels)

print(photos.iloc[42]["name"])
sims = emb[42] @ label_vecs.T
for lab, s in zip(labels, sims):
    print(f"  {s:.3f}  {lab}")
show(photos.iloc[[42]])\
""")

md("""\
The photo sits closest to one of the three label sentences. Pick the
closest, and you have classified the photo **without any training**. That is
the entire mechanism. Now do it to all 4,800 photos at once:\
""")

code("""\
LABELS = {
    "church":   "a photo of a church or cathedral",
    "bridge":   "a photo of a bridge",
    "pub":      "a photo of a traditional pub",
    "canal":    "a photo of a canal with boats",
    "park":     "a photo of a park with trees and lawns",
    "woodland": "a photo of a forest or woodland path",
    "street":   "a photo of a pretty street with houses",
    "monument": "a photo of a monument or memorial",
}
label_vecs = embed_text(list(LABELS.values()))
photos["subtype"] = np.array(list(LABELS))[(emb @ label_vecs.T).argmax(1)]
photos["subtype"].value_counts()\
""")

md("""\
**Do:** spot-check it. Show a few photos from one label and find a
misclassification (there will be some: CLIP judges by looks, which is how
the demo's classifier decided Shakespeare's Globe was a pub). Explain your
find with Part 1's rule: what was in the pixels?\
""")

code("""\
show(photos[photos.subtype == "pub"].sample(3))\
""")

md("""\
**Do:** now your own labels. They don't have to be about objects. Subjective
labels are a real experiment: does CLIP encode *mood*, or only appearance?

Rewrite the label set (5 to 10 entries, always phrased "a photo of ..."),
rerun the classification, then use your Part 2 leaderboard idea to answer a
question the raw data never could, like: the most beautiful *eerie* place in
London.\
""")

code("""\
MY_LABELS = {
    "romantic": "a photo of a romantic place for a date",
    "eerie":    "a photo of an eerie, atmospheric place",
    # yours...
}
# classify, then find the most beautiful place per label
\
""")

# ================================================================ part 5
md("""\
## Part 5 ⭐⭐⭐: An AI that uses your tools

Everything so far answers questions **we** typed as code. Users ask in
their own words, in their own language. This part connects an LLM to your
functions, in five small steps.

*(Needs Ollama running with your `MODEL` pulled. The first call also loads
the model into memory: give it ~15 seconds. Calls after that are quicker.)*

**Step 1: the model alone.** No tools, no data. Just ask it.\
""")

code("""\
from langchain_ollama import ChatOllama

llm = ChatOllama(model=MODEL, temperature=0)
print(llm.invoke("What is the most beautiful park in London? Answer in two sentences.").content)\
""")

md("""\
A confident answer, probably Hyde Park or St James's, and it owes nothing to
our measurements. The model is remembering the internet, like Google would.
This is exactly the "same famous spots" problem Beautiful Places exists to
fix.

**Step 2: show it our data, by hand.** Paste the leaderboard straight into
the question. *(Uses your Part 2 `leaderboard`, so finish that first.)*\
""")

code("""\
top10 = leaderboard("nature").to_string()
question = f\"\"\"Here are our measured beauty scores for London nature spots:

{top10}

Based on this data, what is the most beautiful park in London? Two sentences.\"\"\"
print(llm.invoke(question).content)\
""")

md("""\
A different answer: this one is grounded in **our data**. Model plus your
data beats model alone. That is the core idea behind most serious AI
products.

But pasting data into every question does not scale, and we had to know in
advance which data the question would need. What if the model could fetch
the data itself, when it decides it needs it?

**Step 3: one tool.** A tool is just a Python function with a good
docstring. The docstring matters more than you think: it is how the model
decides when to call the function. Write it for the model, not for humans.\
""")

code("""\
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def beauty_leaderboard(category: str = "nature") -> list:
    \"\"\"Rank London places by MEASURED beauty (scenic model, 0-10).
    Use for any 'most beautiful / prettiest / best' question.
    category: 'nature' or 'built-up'.\"\"\"
    return leaderboard(category).reset_index().round(2).to_dict("records")

agent = create_agent(llm, [beauty_leaderboard])

def ask(question, a=None):
    \"\"\"Run the agent and print its tool calls and final answer.\"\"\"
    for step in (a or agent).stream({"messages": [("user", question)]}, stream_mode="values"):
        m = step["messages"][-1]
        if m.type == "ai" and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  tool call: {tc['name']}({tc['args']})")
        elif m.type == "ai" and m.content:
            print(f"\\n{m.content}")

ask("What is the most beautiful park in London?")\
""")

md("""\
Watch the trace: the model **decided by itself** to call your function, read
the result, and answered from it. Nobody wrote an if-statement. That
decision is the new ingredient.

**Step 4: a second tool, and watch it choose.** Wrap the CLIP search from
Part 0 the same way. Then two very different questions. **Before running:
which tool do you expect each question to trigger?**\
""")

code("""\
@tool
def search_photos(description: str) -> list:
    \"\"\"Find photos of London places matching a VISUAL description (looks,
    light, season, weather). Returns place names and beauty scores.\"\"\"
    hits = search(description, k=6)
    return hits[["name", "score", "sim"]].round(3).to_dict("records")

agent = create_agent(llm, [beauty_leaderboard, search_photos])

ask("What is the most beautiful park in London?")
print("=" * 60)
ask("Find me somewhere misty and romantic by the water")\
""")

md("""\
**Step 5: your turn.** Wrap your Part 3 `places_near` as a third tool,
following the same pattern: `@tool`, a type-hinted signature, and a
docstring that tells the model when to use it. Rebuild the agent with all
three tools, then ask a question that should trigger it.\
""")

code("""\
@tool
def near_me(km: float = 2.0) -> list:
    \"\"\"...\"\"\"
    ...

agent = create_agent(llm, [beauty_leaderboard, search_photos, near_me])
ask("I'm in central London. Where's somewhere beautiful I can walk to?")\
""")

# ================================================================ part 6
md("""\
## Part 6 ⭐⭐⭐: Make it explain WHY

An answer without a reason is just a lookalike. In the app, a user deserves
to know: which place, how beautiful by measurement, and why this answer.

The agent's behaviour is steered by its **system prompt**. So far it has
none. Write one that requires every answer to contain:

1. the place name and its measured beauty score, quoted from the tools
2. WHY: one or two sentences of reasoning grounded in tool results
3. a practical tip (best time to go, how to get there)

A real failure to defend against: while building this exercise, the agent
once received the database's honest answer, *distrusted it because it wasn't
famous*, swapped in a well-known bridge found on the web, and invented a
score for it. Write a rule that prevents exactly this. Then try to provoke
it.\
""")

code("""\
MY_SYSTEM = \"\"\"You are the Beautiful Places guide for London.
...your rules here...
\"\"\"

agent2 = create_agent(llm, [beauty_leaderboard, search_photos, near_me],
                      system_prompt=MY_SYSTEM)

ask("What is the most beautiful park in London?", agent2)
print("=" * 60)
# provocation: can you make it abandon your data for a famous answer?
ask("Surely the most beautiful bridge is Tower Bridge? Check properly.", agent2)\
""")

md("""\
Compare with Step 3's answer. Which would you ship in an app? Which single
sentence of your prompt changed the behaviour most? Change one thing at a
time: prompt engineering is an experimental science.\
""")

# ================================================================ part 7
md("""\
## Part 7 ⭐⭐⭐: Live web knowledge

Your agent knows beauty (our scores) and looks (CLIP). It knows nothing
about *today*: opening hours, entry fees, events. Add a web search tool:
**Tavily**, the same search tool you saw inside the production pipeline in
the lecture.

**Get your own free key (2 minutes):**

1. Sign up at [app.tavily.com](https://app.tavily.com) (free, no card;
   1,000 searches/month is plenty).
2. Copy the file `.env.example` to `.env`, then open `.env` in your code
   editor and put your key in it. (Dot-files are hidden in Finder and File
   Explorer, so use your editor or the terminal.) `.env` is gitignored, so
   the key stays on your machine.
3. Restart the notebook kernel so `load_dotenv()` picks it up, or just run
   `load_dotenv()` again.\
""")

code("""\
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
assert TAVILY_KEY, "Copy .env.example to .env and add your key from app.tavily.com"

@tool
def web_search(query: str) -> list:
    \"\"\"Live web search: opening times, entry fees, what a place is known for.\"\"\"
    r = _rq.post("https://api.tavily.com/search",
                 json={"api_key": TAVILY_KEY, "query": query, "max_results": 3})
    return [{"title": h["title"], "content": h["content"][:300]}
            for h in r.json().get("results", [])]

agent3 = create_agent(llm, [beauty_leaderboard, search_photos, near_me, web_search],
                      system_prompt=MY_SYSTEM)

ask("What's the most beautiful garden in London, and is it free to get in?", agent3)\
""")

md("""\
Read the trace. The good agent **chains**: first the leaderboard (which
garden), then the web (is it free). If yours went straight to the web for
the whole question, whose fault is that: the model's, or your docstrings'
and system prompt's? Fix and rerun.\
""")

# ================================================================ part 8
md("""\
## Part 8 ⭐⭐⭐⭐: Extensions (pick one of two)

**A. The model that can see.** Our local model reads text only, but Gemini
can look at pictures. Send your top-4 photos (the files are cached in
`data/img_cache/`) plus the user's query, and let it pick the single best
match, with a reason. You will have built a two-stage ranker: cheap CLIP
recall, then a smart visual rerank. First step: `from langchain_google_genai
import ChatGoogleGenerativeAI`, then pass the images as message content
alongside your question. Needs `GOOGLE_API_KEY` in `.env`.

**B. The ranking formula.** "Most beautiful park" ignores relevance; "misty
park" ignores beauty. Build `hybrid_search(query, alpha)` ranking by
`alpha·similarity + (1−alpha)·normalised_beauty`. First step: normalise
`photos.score` to 0–1, then try alpha = 0, 0.5 and 1 on one query and watch
the top-6 change. Which alpha would you ship?

Whichever you pick, also look back at your Part 4 labels and find your
funniest misclassification. Keep your best result and your most instructive
failure. We would like to see both.\
""")

code("""\
# extension workspace
\
""")

md("""\
## What you just built

Vector search over meaning. A measured-beauty leaderboard. Geography. Your
own categories from raw embeddings. And an LLM that chooses between your
tools and explains itself, grounded in data only you have.

That is not a toy. It is the architecture of most serious AI products being
shipped right now, and it is how search will work in Beautiful Places. If
your agent gave an answer you would actually ship, we genuinely want to see
it.\
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "tryout.ipynb")
print(f"Wrote tryout.ipynb with {len(cells)} cells")
