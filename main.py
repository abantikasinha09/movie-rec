import os
import json
import asyncio
import subprocess
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx"
    )


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Recommender API",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    return f"{TMDB_IMG_500}{path}"


# =========================
# TMDB GET USING CURL
# =========================
async def tmdb_get(
    path: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:

    url = f"{TMDB_BASE}{path}"

    # Windows -> curl.exe
    # Render/Linux -> curl
    curl_command = "curl.exe" if os.name == "nt" else "curl"

    command = [
        curl_command,
        "-s",
        "--fail-with-body",
        "--get",
        url,
        "--data-urlencode",
        f"api_key={TMDB_API_KEY}",
    ]

    for key, value in params.items():
        if value is not None:
            command.extend([
                "--data-urlencode",
                f"{key}={value}"
            ])

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=502,
            detail="TMDB request timed out"
        )

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB curl error: {type(e).__name__} | {str(e)}"
        )

    if result.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB curl error: {result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="TMDB returned invalid JSON"
        )


# =========================
# TMDB CARD HELPERS
# =========================
async def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20
) -> List[TMDBMovieCard]:

    out: List[TMDBMovieCard] = []

    for m in (results or [])[:limit]:

        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(
                    m.get("poster_path")
                ),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )

    return out


async def tmdb_movie_details(
    movie_id: int
) -> TMDBMovieDetails:

    data = await tmdb_get(
        f"/movie/{movie_id}",
        {
            "language": "en-US"
        }
    )

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(
            data.get("poster_path")
        ),
        backdrop_url=make_img_url(
            data.get("backdrop_path")
        ),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(
    query: str,
    page: int = 1
) -> Dict[str, Any]:

    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(
    query: str
) -> Optional[dict]:

    data = await tmdb_search_movies(
        query=query,
        page=1
    )

    results = data.get("results", [])

    return results[0] if results else None


# =========================
# TF-IDF HELPERS
# =========================
def build_title_to_idx_map(
    indices: Any
) -> Dict[str, int]:

    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):

        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)

        return title_to_idx

    try:

        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)

        return title_to_idx

    except Exception:

        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like "
            "(with .items())"
        )


def get_local_idx_by_title(
    title: str
) -> int:

    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF index map not initialized"
        )

    key = _norm_title(title)

    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])

    raise HTTPException(
        status_code=404,
        detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
) -> List[Tuple[str, float]]:

    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources not loaded"
        )

    idx = get_local_idx_by_title(query_title)

    qv = tfidf_matrix[idx]

    scores = (
        tfidf_matrix @ qv.T
    ).toarray().ravel()

    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []

    for i in order:

        if int(i) == int(idx):
            continue

        try:

            title_i = str(
                df.iloc[int(i)]["title"]
            )

        except Exception:

            continue

        out.append(
            (
                title_i,
                float(scores[int(i)])
            )
        )

        if len(out) >= top_n:
            break

    return out


async def attach_tmdb_card_by_title(
    title: str
) -> Optional[TMDBMovieCard]:

    try:

        m = await tmdb_search_first(title)

        if not m:
            return None

        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(
                m.get("poster_path")
            ),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )

    except Exception:

        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():

    global df
    global indices_obj
    global tfidf_matrix
    global tfidf_obj
    global TITLE_TO_IDX

    # Load dataframe
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load TF-IDF vectorizer
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build title index
    TITLE_TO_IDX = build_title_to_idx_map(
        indices_obj
    )

    # Sanity check
    if df is None or "title" not in df.columns:

        raise RuntimeError(
            "df.pkl must contain a DataFrame "
            "with a 'title' column"
        )


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================
# HOME FEED
# =========================
@app.get(
    "/home",
    response_model=List[TMDBMovieCard]
)
async def home(
    category: str = Query("popular"),
    limit: int = Query(
        24,
        ge=1,
        le=50
    ),
):

    try:

        # Trending
        if category == "trending":

            data = await tmdb_get(
                "/trending/movie/day",
                {
                    "language": "en-US"
                }
            )

            return await tmdb_cards_from_results(
                data.get("results", []),
                limit=limit
            )

        # Other categories
        if category not in {
            "popular",
            "top_rated",
            "upcoming",
            "now_playing"
        }:

            raise HTTPException(
                status_code=400,
                detail="Invalid category"
            )

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1
            }
        )

        return await tmdb_cards_from_results(
            data.get("results", []),
            limit=limit
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Home route failed: {e}"
        )


# =========================
# TMDB SEARCH
# =========================
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(
        ...,
        min_length=1
    ),
    page: int = Query(
        1,
        ge=1,
        le=10
    ),
):

    return await tmdb_search_movies(
        query=query,
        page=page
    )


# =========================
# MOVIE DETAILS
# =========================
@app.get(
    "/movie/id/{tmdb_id}",
    response_model=TMDBMovieDetails
)
async def movie_details_route(
    tmdb_id: int
):

    return await tmdb_movie_details(
        tmdb_id
    )


# =========================
# GENRE RECOMMENDATIONS
# =========================
@app.get("/recommend/genre")
async def recommend_genre(
    genre: str,
    limit: int = 10
):

    try:

        genre = genre.strip().lower()

        if not genre:
            return []

        mask = (
            df["genres"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                genre,
                regex=False
            )
        )

        matches = df[mask].copy()

        if matches.empty:
            return []

        matches = matches.head(limit)

        results = []

        for _, row in matches.iterrows():

            title = row.get("title")

            if not title:
                continue

            results.append(
                {
                    "title": str(title),
                    "poster_url": None,
                    "score": 0.0
                }
            )

        return results

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Genre recommendation error: {str(e)}"
        )


# =========================
# TF-IDF RECOMMENDATIONS
# =========================
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(
        ...,
        min_length=1
    ),
    top_n: int = Query(
        10,
        ge=1,
        le=50
    ),
):

    recs = tfidf_recommend_titles(
        title,
        top_n=top_n
    )

    return [
        {
            "title": t,
            "score": s
        }
        for t, s in recs
    ]


# =========================
# BUNDLE
# =========================
@app.get(
    "/movie/search",
    response_model=SearchBundleResponse
)
async def search_bundle(
    query: str = Query(
        ...,
        min_length=1
    ),
    tfidf_top_n: int = Query(
        12,
        ge=1,
        le=30
    ),
    genre_limit: int = Query(
        12,
        ge=1,
        le=30
    ),
):

    # Find movie on TMDB
    best = await tmdb_search_first(query)

    if not best:

        raise HTTPException(
            status_code=404,
            detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])

    # Movie details
    details = await tmdb_movie_details(
        tmdb_id
    )

    # =========================
    # TF-IDF RECOMMENDATIONS
    # =========================
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []

    try:

        recs = tfidf_recommend_titles(
            details.title,
            top_n=tfidf_top_n
        )

    except Exception:

        try:

            recs = tfidf_recommend_titles(
                query,
                top_n=tfidf_top_n
            )

        except Exception:

            recs = []

    for title, score in recs:

        card = await attach_tmdb_card_by_title(
            title
        )

        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                tmdb=card
            )
        )

    # =========================
    # GENRE RECOMMENDATIONS
    # =========================
    genre_recs: List[TMDBMovieCard] = []

    if details.genres:

        genre_id = details.genres[0]["id"]

        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            }
        )

        cards = await tmdb_cards_from_results(
            discover.get("results", []),
            limit=genre_limit
        )

        genre_recs = [
            c
            for c in cards
            if c.tmdb_id != details.tmdb_id
        ]

    # =========================
    # RETURN BUNDLE
    # =========================
    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
