"""Knowledge base loading and BM25 keyword search (no external dependencies)."""
import math
import re
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "i", "if", "in", "is", "it", "its", "me", "my", "not", "of", "on", "or", "so", "that",
    "the", "to", "up", "was", "when", "with", "can", "cant", "t", "s", "do", "does", "doesn",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in STOPWORDS:
            continue
        # naive stemming so "disconnecting" and "disconnect" match
        for suffix in ("ing", "ed", "es", "s"):
            if len(tok) > 4 and tok.endswith(suffix):
                tok = tok[: -len(suffix)]
                break
        tokens.append(tok)
    return tokens


@dataclass
class Article:
    id: str
    title: str
    team: str
    body: str


def load_articles(kb_dir: str | Path) -> list[Article]:
    articles = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        _, front, body = raw.split("---", 2)
        meta = {}
        for line in front.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        articles.append(Article(meta["id"], meta["title"], meta["team"], body.strip()))
    return articles


class KnowledgeBase:
    """BM25 ranking over whole articles (title is weighted by repeating it)."""

    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75):
        self.articles = articles
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{a.title} {a.title} {a.body}") for a in articles]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.n
        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _score(self, query_tokens: list[str], idx: int) -> float:
        doc = self.docs[idx]
        score = 0.0
        for term in query_tokens:
            tf = doc.count(term)
            if tf == 0:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
        return score

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = tokenize(query)
        ranked = sorted(
            ((self._score(q, i), i) for i in range(self.n)), key=lambda x: x[0], reverse=True
        )
        results = []
        for score, i in ranked[:k]:
            if score <= 0:
                continue
            a = self.articles[i]
            results.append(
                {"id": a.id, "title": a.title, "team": a.team, "score": round(score, 3), "text": a.body}
            )
        return results

    def ids(self) -> set[str]:
        return {a.id for a in self.articles}
