"""
CineMatch AI - Professional Movie Recommendation System
Fully Interactive Netflix-Style UI with Safe Poster Loading
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.tmdb_api import TMDBAPI
from datetime import datetime
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="CineMatch AI - Netflix Style Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: #0a0a0a;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .hero-container {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 30%, #16213e 60%, #0a0a0a 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 40%, #4facfe 70%, #43e97b 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease-in-out infinite;
        position: relative;
        z-index: 1;
        line-height: 1.1;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        letter-spacing: 3px;
    }
    
    .hero-subtitle span {
        color: #f5576c;
        font-weight: 600;
    }
    
    .hero-stats {
        display: flex;
        gap: 2rem;
        margin-top: 1.5rem;
        position: relative;
        z-index: 1;
        flex-wrap: wrap;
    }
    
    .hero-stat {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .hero-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f5576c;
    }
    
    .hero-stat-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-title .emoji {
        font-size: 2rem;
    }
    
    .movie-item {
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        background: rgba(255,255,255,0.03);
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .movie-item:hover {
        transform: scale(1.05);
        z-index: 10;
        box-shadow: 0 15px 50px rgba(0,0,0,0.8);
        border-color: #f5576c;
    }
    
    .movie-item img {
        width: 100%;
        height: 280px;
        object-fit: cover;
        border-radius: 10px;
    }
    
    .movie-title-text {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        margin-top: 0.7rem !important;
        text-align: center !important;
        display: block !important;
        padding: 0.3rem !important;
        background: rgba(0,0,0,0.5) !important;
        border-radius: 8px !important;
        min-height: 2.5rem !important;
        line-height: 1.3 !important;
    }
    
    .movie-title-text:hover {
        color: #f5576c !important;
    }
    
    .poster-placeholder {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 280px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        color: white;
        font-size: 4rem;
    }
    
    .rec-placeholder {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: white;
        font-size: 3rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.05);
        padding: 0.3rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 2rem;
        border-radius: 12px;
        color: rgba(255,255,255,0.5);
        font-weight: 600;
        background: transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: white;
        background: rgba(255,255,255,0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        box-shadow: 0 5px 20px rgba(245, 87, 108, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 40px rgba(245, 87, 108, 0.5);
        color: white;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        border-radius: 50px !important;
        color: white !important;
        padding: 1.2rem 2rem !important;
        font-size: 1.1rem !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #f5576c !important;
        box-shadow: 0 0 50px rgba(245, 87, 108, 0.15) !important;
        background: rgba(255,255,255,0.08) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.3) !important;
    }
    
    .rec-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.4s ease;
        margin-bottom: 1rem;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        border-color: #f5576c;
        box-shadow: 0 10px 40px rgba(245, 87, 108, 0.15);
    }
    
    .rec-title {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        margin-top: 0.5rem !important;
    }
    
    .rec-genre {
        color: rgba(255,255,255,0.5) !important;
        font-size: 0.9rem !important;
    }
    
    .rec-score {
        color: #ffd700 !important;
        font-weight: 600 !important;
    }
    
    .footer {
        text-align: center;
        padding: 3rem 2rem 1rem 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }
    
    .footer-text {
        color: rgba(255,255,255,0.2);
        font-size: 0.8rem;
        letter-spacing: 1px;
    }
    
    .footer-text span {
        color: #f5576c;
    }
    
    .search-results-title {
        color: white !important;
        font-size: 1.5rem !important;
        margin: 1.5rem 0 !important;
    }
    
    .search-results-title span {
        color: #f5576c !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_resource
def load_all_data():
    loader = MovieDataLoader()
    loader.load_all()
    content = ContentBasedRecommender()
    collab = CollaborativeRecommender()
    hybrid = HybridRecommender()
    tmdb = TMDBAPI()
    return loader, content, collab, hybrid, tmdb

with st.spinner("🎬 Loading CineMatch AI..."):
    loader, content_rec, collab_rec, hybrid_rec, tmdb = load_all_data()

# ==================== SESSION STATE ====================
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# ==================== SAFE POSTER LOADER ====================
def safe_get_poster(movie_title):
    """Safely get poster with error handling"""
    try:
        poster_url, details = tmdb.search_and_get_poster(movie_title)
        return poster_url, details
    except Exception as e:
        return None, None

# ==================== HERO SECTION ====================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">CineMatch AI</div>
    <div class="hero-subtitle">Discover movies you'll love with <span>AI-powered</span> recommendations</div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">{loader.get_movie_count():,}</div>
            <div class="hero-stat-label">Movies</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">{len(loader.ratings):,}</div>
            <div class="hero-stat-label">Ratings</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">{loader.get_user_count():,}</div>
            <div class="hero-stat-label">Users</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">⭐ {loader.get_rating_stats()['mean']:.2f}</div>
            <div class="hero-stat-label">Avg Rating</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SEARCH ====================
st.markdown("### 🔍 Find Your Next Movie")
search_query = st.text_input(
    "Search for a movie",
    placeholder="e.g., Inception, The Matrix, Titanic",
    label_visibility="collapsed",
    key="search_input"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("🔍 Search", width="stretch")

# ==================== SHOW SEARCH RESULTS ====================
if search_clicked and search_query:
    st.markdown(f"""
    <div style="margin: 2rem 0;">
        <h2 class="search-results-title">🔍 Results for: <span>{search_query}</span></h2>
    </div>
    """, unsafe_allow_html=True)
    
    movies = content_rec.search_movies(search_query)
    if len(movies) > 0:
        st.success(f"Found {len(movies)} matching movies!")
        
        for i in range(0, min(len(movies), 12), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(movies):
                    movie_title = movies.iloc[idx]['title']
                    
                    with col:
                        poster_url, _ = safe_get_poster(movie_title)
                        
                        st.markdown(f"""
                        <div class="movie-item">
                        """, unsafe_allow_html=True)
                        
                        if poster_url:
                            st.image(poster_url, width="stretch")
                        else:
                            st.markdown(f"""
                            <div class="poster-placeholder">🎬</div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="movie-title-text">{movie_title}</div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"🎯 Get Recs", key=f"search_rec_{idx}_{j}"):
                            st.session_state.selected_movie = movie_title
                            st.session_state.show_recommendations = True
                            st.rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No movies found. Try a different search term.")

# ==================== SHOW RECOMMENDATIONS ====================
if st.session_state.selected_movie and st.session_state.show_recommendations:
    st.markdown("---")
    
    with st.spinner(f"Finding movies like {st.session_state.selected_movie}..."):
        movies = content_rec.search_movies(st.session_state.selected_movie)
        if len(movies) > 0:
            selected = movies.iloc[0]['title']
            recommendations = content_rec.get_recommendations(selected, n=12)
            
            st.markdown(f"""
            <div class="section-header">
                <div class="section-title">🎯 Movies like <span style="color: #f5576c;">{selected}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            if recommendations is not None and len(recommendations) > 0:
                for i in range(0, len(recommendations), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(recommendations):
                            row = recommendations.iloc[idx]
                            rec_title = row['title']
                            rec_genres = row['genres']
                            rec_score = row['similarity_score']
                            
                            poster_url, details = safe_get_poster(rec_title)
                            
                            with col:
                                st.markdown(f"""
                                <div class="rec-card">
                                """, unsafe_allow_html=True)
                                
                                if poster_url:
                                    st.image(poster_url, width="stretch")
                                else:
                                    st.markdown(f"""
                                    <div class="rec-placeholder">🎬</div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                    <div class="rec-title">{rec_title}</div>
                                    <div class="rec-genre">🎭 {rec_genres}</div>
                                    <div class="rec-score">⭐ Similarity: {rec_score:.3f}</div>
                                """, unsafe_allow_html=True)
                                
                                if details and details.get('overview'):
                                    with st.expander("📝 Synopsis"):
                                        st.write(details['overview'][:200] + "...")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("← Back to Home", width="stretch", key="back_btn"):
                st.session_state.selected_movie = None
                st.session_state.show_recommendations = False
                st.rerun()

# ==================== TABS ====================
if not st.session_state.selected_movie or not st.session_state.show_recommendations:
    
    # ==================== MOVIE DATA ====================
    trending_movies = [
        "Inception", "The Dark Knight", "Interstellar", "The Matrix",
        "Pulp Fiction", "Forrest Gump", "Titanic", "The Godfather",
        "The Shawshank Redemption", "Fight Club", "The Avengers", "Avatar"
    ]
    
    for_you_movies = [
        "The Matrix", "Inception", "The Dark Knight", "Interstellar",
        "Star Wars", "Jurassic Park", "Back to the Future", "The Terminator",
        "Alien", "Predator", "RoboCop", "Total Recall"
    ]
    
    top_rated_movies = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight",
        "Pulp Fiction", "Forrest Gump", "Inception", "The Matrix",
        "Fight Club", "Interstellar", "The Silence of the Lambs"
    ]
    
    def display_movie_row(movies, title, emoji="🎬"):
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title"><span class="emoji">{emoji}</span> {title}</div>
        </div>
        """, unsafe_allow_html=True)
        
        for i in range(0, min(len(movies), 12), 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(movies):
                    movie_title = movies[idx]
                    
                    with col:
                        poster_url, _ = safe_get_poster(movie_title)
                        
                        st.markdown(f"""
                        <div class="movie-item">
                        """, unsafe_allow_html=True)
                        
                        if poster_url:
                            st.image(poster_url, width="stretch")
                        else:
                            st.markdown(f"""
                            <div class="poster-placeholder">🎬</div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="movie-title-text">{movie_title}</div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"🎯 Get Recs", key=f"row_{title}_{idx}_{j}"):
                            st.session_state.selected_movie = movie_title
                            st.session_state.show_recommendations = True
                            st.rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)

    # ==================== CREATE 5 TABS ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Trending", "🎯 For You", "⭐ Top Rated", "🎭 Genres", "📊 Analytics"])

    # ==================== TAB 1: TRENDING ====================
    with tab1:
        display_movie_row(trending_movies[:8], "🔥 Trending Now", "🔥")
        display_movie_row(trending_movies[8:], "📈 Popular This Week", "📈")

    # ==================== TAB 2: FOR YOU ====================
    with tab2:
        display_movie_row(for_you_movies[:8], "🎯 Recommended For You", "🎯")
        display_movie_row(for_you_movies[8:], "👀 Because You Watched...", "👀")

    # ==================== TAB 3: TOP RATED ====================
    with tab3:
        display_movie_row(top_rated_movies[:6], "⭐ Top Rated Movies", "⭐")
        display_movie_row(top_rated_movies[6:], "🏆 Critics' Choice", "🏆")

    # ==================== TAB 4: GENRES ====================
    with tab4:
        genre_movies = [
            ("💥 Action", ["The Dark Knight", "Inception", "The Matrix", "Interstellar", "Gladiator", "Die Hard"]),
            ("😂 Comedy", ["The Hangover", "Superbad", "Bridesmaids", "Anchorman", "Step Brothers", "Old School"]),
            ("🎭 Drama", ["The Shawshank Redemption", "The Godfather", "Forrest Gump", "Fight Club", "The Green Mile", "Goodfellas"])
        ]
        for genre_name, movies_list in genre_movies:
            display_movie_row(movies_list, genre_name, "🎭")

    # ==================== TAB 5: ANALYTICS ====================
    with tab5:
        st.markdown("""
        <div style="margin: 1rem 0 2rem 0;">
            <h2 style="color: white;">📊 Analytics Dashboard</h2>
            <p style="color: rgba(255,255,255,0.5);">Explore insights from the MovieLens dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        data = loader
        
        # Dataset Overview
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Movies", data.get_movie_count())
        with col2:
            st.metric("⭐ Ratings", f"{len(data.ratings):,}")
        with col3:
            st.metric("👤 Users", data.get_user_count())
        with col4:
            st.metric("📊 Avg Rating", f"{data.get_rating_stats()['mean']:.2f}")
        
        st.markdown("---")
        
        # Rating Statistics
        st.markdown("### ⭐ Rating Statistics")
        stats = data.get_rating_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{stats['mean']:.2f}/5")
            st.metric("Median", f"{stats['median']:.2f}/5")
        with col2:
            st.metric("Min", f"{stats['min']}/5")
            st.metric("Max", f"{stats['max']}/5")
        with col3:
            st.metric("Standard Deviation", f"{stats['std']:.3f}")
        
        st.markdown("---")
        
        # Genre Distribution
        st.markdown("### 🎭 Genre Distribution")
        
        all_genres = []
        for genres in data.movies['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        
        # Display top genres in a dataframe
        genre_df = genre_counts.reset_index()
        genre_df.columns = ['Genre', 'Count']
        st.dataframe(genre_df.head(10), use_container_width=True)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#f5576c', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#a18cd1', '#fbc2eb']
        genre_counts.head(10).plot(kind='bar', ax=ax, color=colors[:len(genre_counts.head(10))])
        ax.set_title('Top 10 Most Common Genres', color='white', fontsize=14)
        ax.set_xlabel('Genre', color='white')
        ax.set_ylabel('Number of Movies', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Most Popular Movies
        st.markdown("### 🏆 Most Popular Movies")
        
        movie_ratings = data.ratings.groupby('movieId')['rating'].count()
        movie_ratings = movie_ratings.sort_values(ascending=False)
        top_movies = movie_ratings.head(10)
        
        for i, (movie_id, count) in enumerate(top_movies.items(), 1):
            title = data.get_movie_title(movie_id)
            if title:
                avg_rating = data.ratings[data.ratings['movieId'] == movie_id]['rating'].mean()
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); 
                            padding: 0.7rem 1rem; 
                            border-radius: 10px;
                            margin: 0.3rem 0;
                            border-left: 3px solid #f5576c;
                            display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: white; font-weight: 600;">#{i} {title}</span>
                    <span style="color: rgba(255,255,255,0.5);">
                        ⭐ {avg_rating:.2f} · {count} ratings
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown(f"""
<div class="footer">
    <div class="footer-text">
        Built with <span>❤️</span> using Streamlit · MovieLens Dataset · TMDB API<br>
        © {datetime.now().year} CineMatch AI — Your Personal Movie Companion
    </div>
</div>
""", unsafe_allow_html=True)