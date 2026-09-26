import os
import json
import pickle
import asyncio
import subprocess
from urllib.parse import urlencode
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================================================
# ENV
# =========================================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx"
    )


# =========================================================
# FASTAPI APP
# =========================================================

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


# =========================================================
# PICKLE PATHS
# =========================================================

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


# =========================================================
# MODELS
# =========================================================

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


# =========================================================
# GENERAL UTILS
# =========================================================

def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None

    return f"{TMDB_IMG_500}{path}"


# =========================================================
# TMDB REQUEST
# =========================================================

async def tmdb_get(
    path: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    params = params or {}

    query_params = dict(params)
    query_params["api_key"] = TMDB_API_KEY

    query_string = urlencode(
        query_params,
        doseq=True
    )

    url = f"{TMDB_BASE}{path}?{query_string}"

    try:

        result = await asyncio.to_thread(
            subprocess.run,
            [
                "curl.exe",
                "-L",
                "--fail-with-body",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "20",
                "--max-time",
                "40",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        # -------------------------------------------------
        # CURL FAILED
        # -------------------------------------------------

        if result.returncode != 0:

            error_message = (
                stderr.strip()
                or stdout.strip()
                or f"curl exited with code {result.returncode}"
            )

            raise HTTPException(
                status_code=502,
                detail=f"TMDB curl error: {error_message}"
            )

        # -------------------------------------------------
        # EMPTY RESPONSE
        # -------------------------------------------------

        if not stdout.strip():

            raise HTTPException(
                status_code=502,
                detail="TMDB returned an empty response."
            )

        # -------------------------------------------------
        # JSON RESPONSE
        # -------------------------------------------------

        try:

            data = json.loads(stdout)

        except json.JSONDecodeError:

            raise HTTPException(
                status_code=502,
                detail=(
                    "TMDB returned invalid JSON: "
                    + stdout[:500]
                )
            )

        # -------------------------------------------------
        # MAKE SURE RESPONSE IS A DICT
        # -------------------------------------------------

        if not isinstance(data, dict):

            raise HTTPException(
                status_code=502,
                detail="TMDB returned an unexpected response format."
            )

        # -------------------------------------------------
        # TMDB ERROR RESPONSE
        # -------------------------------------------------

        if "status_code" in data and data.get("status_code") not in (0, None):

            raise HTTPException(
                status_code=502,
                detail=(
                    f"TMDB API error "
                    f"{data.get('status_code')}: "
                    f"{data.get('status_message', 'Unknown error')}"
                )
            )

        return data

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=(
                f"TMDB request error: "
                f"{type(e).__name__}: {e}"
            )
        )


# =========================================================
# TMDB MOVIE CARDS
# =========================================================

async def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20
) -> List[TMDBMovieCard]:

    output: List[TMDBMovieCard] = []

    for movie in (results or [])[:limit]:

        try:

            output.append(
                TMDBMovieCard(
                    tmdb_id=int(movie["id"]),
                    title=(
                        movie.get("title")
                        or movie.get("name")
                        or ""
                    ),
                    poster_url=make_img_url(
                        movie.get("poster_path")
                    ),
                    release_date=movie.get(
                        "release_date"
                    ),
                    vote_average=movie.get(
                        "vote_average"
                    ),
                )
            )

        except Exception:
            continue

    return output


# =========================================================
# MOVIE DETAILS
# =========================================================

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

        release_date=data.get(
            "release_date"
        ),

        poster_url=make_img_url(
            data.get("poster_path")
        ),

        backdrop_url=make_img_url(
            data.get("backdrop_path")
        ),

        genres=data.get(
            "genres",
            []
        ) or [],
    )


# =========================================================
# TMDB SEARCH
# =========================================================

async def tmdb_search_movies(
    query: str,
    page: int = 1
) -> Dict[str, Any]:

    query = str(query).strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty."
        )

    data = await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        }
    )

    if "results" not in data:

        data["results"] = []

    return data


async def tmdb_search_first(
    query: str
) -> Optional[dict]:

    data = await tmdb_search_movies(
        query=query,
        page=1
    )

    results = data.get(
        "results",
        []
    )

    if not results:
        return None

    return results[0]


# =========================================================
# TF-IDF HELPERS
# =========================================================

def build_title_to_idx_map(
    indices: Any
) -> Dict[str, int]:

    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):

        for key, value in indices.items():

            title_to_idx[
                _norm_title(key)
            ] = int(value)

        return title_to_idx

    try:

        for key, value in indices.items():

            title_to_idx[
                _norm_title(key)
            ] = int(value)

        return title_to_idx

    except Exception:

        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like."
        )


def get_local_idx_by_title(
    title: str
) -> int:

    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF index map not initialized."
        )

    key = _norm_title(title)

    if key in TITLE_TO_IDX:

        return int(
            TITLE_TO_IDX[key]
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"Title not found in local dataset: "
            f"'{title}'"
        )
    )


def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
) -> List[Tuple[str, float]]:

    global df
    global tfidf_matrix

    if df is None:

        raise HTTPException(
            status_code=500,
            detail="Movie dataframe not loaded."
        )

    if tfidf_matrix is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF matrix not loaded."
        )

    idx = get_local_idx_by_title(
        query_title
    )

    # Query vector
    query_vector = tfidf_matrix[idx]

    # Cosine similarity
    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    # Highest similarity first
    order = np.argsort(-scores)

    recommendations: List[
        Tuple[str, float]
    ] = []

    for i in order:

        if int(i) == int(idx):
            continue

        try:

            movie_title = str(
                df.iloc[int(i)]["title"]
            )

        except Exception:

            continue

        recommendations.append(
            (
                movie_title,
                float(scores[int(i)])
            )
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# =========================================================
# ATTACH TMDB POSTER TO LOCAL MOVIE
# =========================================================

async def attach_tmdb_card_by_title(
    title: str
) -> Optional[TMDBMovieCard]:

    try:

        movie = await tmdb_search_first(
            title
        )

        if not movie:
            return None

        return TMDBMovieCard(

            tmdb_id=int(
                movie["id"]
            ),

            title=(
                movie.get("title")
                or title
            ),

            poster_url=make_img_url(
                movie.get("poster_path")
            ),

            release_date=movie.get(
                "release_date"
            ),

            vote_average=movie.get(
                "vote_average"
            ),
        )

    except Exception:

        return None


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def load_pickles():

    global df
    global indices_obj
    global tfidf_matrix
    global tfidf_obj
    global TITLE_TO_IDX

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    with open(
        DF_PATH,
        "rb"
    ) as file:

        df = pickle.load(file)

    # -----------------------------------------------------
    # INDICES
    # -----------------------------------------------------

    with open(
        INDICES_PATH,
        "rb"
    ) as file:

        indices_obj = pickle.load(file)

    # -----------------------------------------------------
    # TF-IDF MATRIX
    # -----------------------------------------------------

    with open(
        TFIDF_MATRIX_PATH,
        "rb"
    ) as file:

        tfidf_matrix = pickle.load(file)

    # -----------------------------------------------------
    # TF-IDF OBJECT
    # -----------------------------------------------------

    with open(
        TFIDF_PATH,
        "rb"
    ) as file:

        tfidf_obj = pickle.load(file)

    # -----------------------------------------------------
    # TITLE MAP
    # -----------------------------------------------------

    TITLE_TO_IDX = build_title_to_idx_map(
        indices_obj
    )

    # -----------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------

    if (
        df is None
        or "title" not in df.columns
    ):

        raise RuntimeError(
            "df.pkl must contain a DataFrame "
            "with a 'title' column."
        )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# HOME FEED
# =========================================================

@app.get(
    "/home",
    response_model=List[TMDBMovieCard]
)
async def home(

    category: str = Query(
        "popular"
    ),

    limit: int = Query(
        24,
        ge=1,
        le=50
    ),
):

    try:

        # -------------------------------------------------
        # TRENDING
        # -------------------------------------------------

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

        # -------------------------------------------------
        # NORMAL CATEGORIES
        # -------------------------------------------------

        allowed_categories = {
            "popular",
            "top_rated",
            "upcoming",
            "now_playing",
        }

        if category not in allowed_categories:

            raise HTTPException(
                status_code=400,
                detail="Invalid category."
            )

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1,
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


# =========================================================
# TMDB SEARCH
# =========================================================

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


# =========================================================
# MOVIE DETAILS
# =========================================================

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


# =========================================================
# GENRE RECOMMENDATIONS
# =========================================================

@app.get(
    "/recommend/genre",
    response_model=List[TMDBMovieCard]
)
async def recommend_genre(

    tmdb_id: int = Query(...),

    limit: int = Query(
        18,
        ge=1,
        le=50
    ),
):

    details = await tmdb_movie_details(
        tmdb_id
    )

    if not details.genres:

        return []

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
        discover.get(
            "results",
            []
        ),
        limit=limit
    )

    return [
        card
        for card in cards
        if card.tmdb_id != tmdb_id
    ]


# =========================================================
# TF-IDF ONLY
# =========================================================

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

    recommendations = (
        tfidf_recommend_titles(
            title,
            top_n=top_n
        )
    )

    return [
        {
            "title": movie_title,
            "score": score
        }

        for movie_title, score
        in recommendations
    ]


# =========================================================
# MOVIE SEARCH BUNDLE
# =========================================================

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

    # -----------------------------------------------------
    # FIND MOVIE ON TMDB
    # -----------------------------------------------------

    best = await tmdb_search_first(
        query
    )

    if not best:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No TMDB movie found "
                f"for query: {query}"
            )
        )

    tmdb_id = int(
        best["id"]
    )

    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    details = await tmdb_movie_details(
        tmdb_id
    )

    # =====================================================
    # TF-IDF RECOMMENDATIONS
    # =====================================================

    tfidf_items: List[
        TFIDFRecItem
    ] = []

    recommendations: List[
        Tuple[str, float]
    ] = []

    try:

        recommendations = (
            tfidf_recommend_titles(
                details.title,
                top_n=tfidf_top_n
            )
        )

    except Exception:

        try:

            recommendations = (
                tfidf_recommend_titles(
                    query,
                    top_n=tfidf_top_n
                )
            )

        except Exception:

            recommendations = []

    # -----------------------------------------------------
    # ADD TMDB POSTERS
    # -----------------------------------------------------

    for movie_title, score in recommendations:

        card = await attach_tmdb_card_by_title(
            movie_title
        )

        tfidf_items.append(
            TFIDFRecItem(
                title=movie_title,
                score=score,
                tmdb=card
            )
        )

    # =====================================================
    # GENRE RECOMMENDATIONS
    # =====================================================

    genre_recommendations: List[
        TMDBMovieCard
    ] = []

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
            discover.get(
                "results",
                []
            ),
            limit=genre_limit
        )

        genre_recommendations = [
            card
            for card in cards
            if card.tmdb_id != details.tmdb_id
        ]

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return SearchBundleResponse(

        query=query,

        movie_details=details,

        tfidf_recommendations=tfidf_items,

        genre_recommendations=genre_recommendations,
    )
