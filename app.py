import requests
import streamlit as st


# ==========================================================
# CONFIG
# ==========================================================

# Local FastAPI backend
API_BASE = "http://127.0.0.1:8000"

# TMDB image base URL
TMDB_IMG = "https://image.tmdb.org/t/p/w500"


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# ==========================================================
# STYLES
# ==========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }

    .movie-title {
        font-size: 0.9rem;
        line-height: 1.15rem;
        height: 2.3rem;
        overflow: hidden;
    }

    .card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 14px;
        background: rgba(255,255,255,0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# STATE + ROUTING
# ==========================================================

if "view" not in st.session_state:
    st.session_state.view = "home"

if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None


qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")


if qp_view in ("home", "details"):
    st.session_state.view = qp_view


if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except (ValueError, TypeError):
        pass


def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None

    st.query_params["view"] = "home"

    if "id" in st.query_params:
        del st.query_params["id"]

    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)

    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))

    st.rerun()


# ==========================================================
# API HELPERS
# ==========================================================

@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):

    try:
        response = requests.get(
            f"{API_BASE}{path}",
            params=params,
            timeout=25,
        )

        if response.status_code >= 400:
            return None, (
                f"HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, (
            "Could not connect to the FastAPI backend. "
            "Make sure Uvicorn is running on "
            "http://127.0.0.1:8000"
        )

    except requests.exceptions.Timeout:
        return None, "Request timed out."

    except Exception as e:
        return None, f"Request failed: {e}"


# ==========================================================
# POSTER GRID
# ==========================================================

def poster_grid(cards, cols=6, key_prefix="grid"):

    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols

    idx = 0

    for row in range(rows):

        colset = st.columns(cols)

        for col_index in range(cols):

            if idx >= len(cards):
                break

            movie = cards[idx]
            idx += 1

            tmdb_id = movie.get("tmdb_id")
            title = movie.get("title", "Untitled")
            poster = movie.get("poster_url")

            with colset[col_index]:

                if poster:
                    st.image(
                        poster,
                        use_container_width=True,
                    )

                else:
                    st.write("🖼️ No poster")

                if st.button(
                    "Open",
                    key=(
                        f"{key_prefix}_"
                        f"{row}_"
                        f"{col_index}_"
                        f"{idx}_"
                        f"{tmdb_id}"
                    ),
                ):

                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>",
                    unsafe_allow_html=True,
                )


# ==========================================================
# TF-IDF → CARD CONVERTER
# ==========================================================

def to_cards_from_tfidf_items(tfidf_items):

    cards = []

    for item in tfidf_items or []:

        tmdb = item.get("tmdb") or {}

        if tmdb.get("tmdb_id"):

            cards.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": (
                        tmdb.get("title")
                        or item.get("title")
                        or "Untitled"
                    ),
                    "poster_url": tmdb.get("poster_url"),
                }
            )

    return cards


# ==========================================================
# TMDB SEARCH PARSER
# ==========================================================

def parse_tmdb_search_to_cards(
    data,
    keyword: str,
    limit: int = 24,
):
    """
    Supports both API formats:

    1. Raw TMDB:
       {
           "results": [
               {
                   "id": ...,
                   "title": ...,
                   "poster_path": ...
               }
           ]
       }

    2. Already processed cards:
       [
           {
               "tmdb_id": ...,
               "title": ...,
               "poster_url": ...
           }
       ]
    """

    suggestions = []
    cards = []

    keyword_l = keyword.strip().lower()

    raw_items = []

    # ------------------------------------------------------
    # FORMAT A: Dictionary containing "results"
    # ------------------------------------------------------

    if isinstance(data, dict) and "results" in data:

        raw = data.get("results") or []

        for movie in raw:

            title = (movie.get("title") or "").strip()
            tmdb_id = movie.get("id")
            poster_path = movie.get("poster_path")

            if not title or not tmdb_id:
                continue

            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": (
                        f"{TMDB_IMG}{poster_path}"
                        if poster_path
                        else None
                    ),
                    "release_date": movie.get(
                        "release_date",
                        "",
                    ),
                }
            )

    # ------------------------------------------------------
    # FORMAT B: Already processed list
    # ------------------------------------------------------

    elif isinstance(data, list):

        for movie in data:

            tmdb_id = (
                movie.get("tmdb_id")
                or movie.get("id")
            )

            title = (
                movie.get("title") or ""
            ).strip()

            poster_url = movie.get("poster_url")

            if not title or not tmdb_id:
                continue

            raw_items.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "title": title,
                    "poster_url": poster_url,
                    "release_date": movie.get(
                        "release_date",
                        "",
                    ),
                }
            )

    else:
        return [], []

    # ------------------------------------------------------
    # WORD-MATCH FILTER
    # ------------------------------------------------------

    matched = [
        movie
        for movie in raw_items
        if keyword_l in movie["title"].lower()
    ]

    # If no direct match, show raw results
    final_list = matched if matched else raw_items

    # ------------------------------------------------------
    # SUGGESTIONS
    # ------------------------------------------------------

    for movie in final_list[:10]:

        year = (
            movie.get("release_date") or ""
        )[:4]

        if year:
            label = f"{movie['title']} ({year})"
        else:
            label = movie["title"]

        suggestions.append(
            (
                label,
                movie["tmdb_id"],
            )
        )

    # ------------------------------------------------------
    # CARDS
    # ------------------------------------------------------

    cards = [
        {
            "tmdb_id": movie["tmdb_id"],
            "title": movie["title"],
            "poster_url": movie["poster_url"],
        }
        for movie in final_list[:limit]
    ]

    return suggestions, cards


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🎬 Menu")

    if st.button("🏠 Home"):
        goto_home()

    st.markdown("---")

    st.markdown("### 🏠 Home Feed")

    home_category = st.selectbox(
        "Category",
        [
            "trending",
            "popular",
            "top_rated",
            "now_playing",
            "upcoming",
        ],
        index=0,
    )

    grid_cols = st.slider(
        "Grid columns",
        4,
        8,
        6,
    )


# ==========================================================
# HEADER
# ==========================================================

st.title("🎬 Movie Recommender")

st.markdown(
    """
    <div class='small-muted'>
    Type keyword → dropdown suggestions + matching results
    → open → details + recommendations
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ==========================================================
# VIEW: HOME
# ==========================================================

if st.session_state.view == "home":

    typed = st.text_input(
        "Search by movie title (keyword)",
        placeholder="Type: avenger, batman, love...",
    )

    st.divider()

    # ======================================================
    # SEARCH MODE
    # ======================================================

    if typed.strip():

        if len(typed.strip()) < 2:

            st.caption(
                "Type at least 2 characters for suggestions."
            )

        else:

            data, error = api_get_json(
                "/tmdb/search",
                params={
                    "query": typed.strip()
                },
            )

            if error or data is None:

                st.error(
                    f"Search failed: {error}"
                )

            else:

                suggestions, cards = (
                    parse_tmdb_search_to_cards(
                        data,
                        typed.strip(),
                        limit=24,
                    )
                )

                # ------------------------------------------------
                # DROPDOWN
                # ------------------------------------------------

                if suggestions:

                    labels = (
                        ["-- Select a movie --"]
                        + [
                            suggestion[0]
                            for suggestion in suggestions
                        ]
                    )

                    selected = st.selectbox(
                        "Suggestions",
                        labels,
                        index=0,
                    )

                    if selected != "-- Select a movie --":

                        label_to_id = {
                            suggestion[0]: suggestion[1]
                            for suggestion in suggestions
                        }

                        goto_details(
                            label_to_id[selected]
                        )

                else:

                    st.info(
                        "No suggestions found. "
                        "Try another keyword."
                    )

                # ------------------------------------------------
                # SEARCH RESULTS
                # ------------------------------------------------

                st.markdown("### Results")

                poster_grid(
                    cards,
                    cols=grid_cols,
                    key_prefix="search_results",
                )

        st.stop()

    # ======================================================
    # HOME FEED
    # ======================================================

    st.markdown(
        f"### 🏠 Home — "
        f"{home_category.replace('_', ' ').title()}"
    )

    home_cards, error = api_get_json(
        "/home",
        params={
            "category": home_category,
            "limit": 24,
        },
    )

    if error or not home_cards:

        st.error(
            f"Home feed failed: "
            f"{error or 'Unknown error'}"
        )

        st.stop()

    poster_grid(
        home_cards,
        cols=grid_cols,
        key_prefix="home_feed",
    )


# ==========================================================
# VIEW: DETAILS
# ==========================================================

elif st.session_state.view == "details":

    tmdb_id = st.session_state.selected_tmdb_id

    if not tmdb_id:

        st.warning("No movie selected.")

        if st.button("← Back to Home"):
            goto_home()

        st.stop()

    # ======================================================
    # TOP BAR
    # ======================================================

    left_top, right_top = st.columns(
        [3, 1]
    )

    with left_top:

        st.markdown(
            "### 📄 Movie Details"
        )

    with right_top:

        if st.button("← Back to Home"):

            goto_home()

    # ======================================================
    # MOVIE DETAILS
    # ======================================================

    data, error = api_get_json(
        f"/movie/id/{tmdb_id}"
    )

    if error or not data:

        st.error(
            f"Could not load details: "
            f"{error or 'Unknown error'}"
        )

        st.stop()

    # ======================================================
    # POSTER + DETAILS
    # ======================================================

    left, right = st.columns(
        [1, 2.4],
        gap="large",
    )

    with left:

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True,
        )

        if data.get("poster_url"):

            st.image(
                data["poster_url"],
                use_container_width=True,
            )

        else:

            st.write("🖼️ No poster")

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"## {data.get('title', '')}"
        )

        release = (
            data.get("release_date")
            or "-"
        )

        genres = ", ".join(
            [
                genre["name"]
                for genre in data.get(
                    "genres",
                    [],
                )
                if genre.get("name")
            ]
        ) or "-"

        st.markdown(
            f"""
            <div class='small-muted'>
            Release: {release}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class='small-muted'>
            Genres: {genres}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown("### Overview")

        st.write(
            data.get("overview")
            or "No overview available."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # ======================================================
    # BACKDROP
    # ======================================================

    if data.get("backdrop_url"):

        st.markdown("#### Backdrop")

        st.image(
            data["backdrop_url"],
            use_container_width=True,
        )

    # ======================================================
    # RECOMMENDATIONS
    # ======================================================

    st.divider()

    st.markdown(
        "### ✅ Recommendations"
    )

    title = (
        data.get("title") or ""
    ).strip()

    if title:

        bundle, error2 = api_get_json(
            "/movie/search",
            params={
                "query": title,
                "tfidf_top_n": 12,
                "genre_limit": 12,
            },
        )

        # ==================================================
        # TF-IDF + GENRE
        # ==================================================

        if not error2 and bundle:

            st.markdown(
                "#### 🔎 Similar Movies (TF-IDF)"
            )

            tfidf_cards = (
                to_cards_from_tfidf_items(
                    bundle.get(
                        "tfidf_recommendations"
                    )
                )
            )

            poster_grid(
                tfidf_cards,
                cols=grid_cols,
                key_prefix="details_tfidf",
            )

            st.markdown(
                "#### 🎭 More Like This (Genre)"
            )

            poster_grid(
                bundle.get(
                    "genre_recommendations",
                    [],
                ),
                cols=grid_cols,
                key_prefix="details_genre",
            )

        # ==================================================
        # FALLBACK: GENRE ONLY
        # ==================================================

        else:

            st.info(
                "Showing Genre recommendations "
                "(fallback)."
            )

            genre_only, error3 = api_get_json(
                "/recommend/genre",
                params={
                    "tmdb_id": tmdb_id,
                    "limit": 18,
                },
            )

            if not error3 and genre_only:

                poster_grid(
                    genre_only,
                    cols=grid_cols,
                    key_prefix=(
                        "details_genre_fallback"
                    ),
                )

            else:

                st.warning(
                    "No recommendations "
                    "available right now."
                )

    else:

        st.warning(
            "No title available to compute "
            "recommendations."
        )