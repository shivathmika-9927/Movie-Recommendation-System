"""
Content-Based Recommendation System for CineMatch AI
Recommends movies based on similarity of genres and tags
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import MovieDataLoader
import pickle
import os

class ContentBasedRecommender:
    """
    Content-based recommender using movie genres and tags
    """
    
    def __init__(self, force_rebuild=False):
        """
        Initialize the recommender with data
        
        Args:
            force_rebuild (bool): Force rebuild similarity matrix
        """
        self.loader = MovieDataLoader()
        self.movies, self.ratings, self.tags = self.loader.load_all()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
        # Try to load existing model, or build new one
        if not force_rebuild:
            try:
                self.load_model()
                print("✅ Loaded existing content-based model")
            except:
                print("⚠️ No existing model found. Building new one...")
                self.build_feature_matrix()
                self.calculate_similarity()
                self.save_model()
        else:
            self.build_feature_matrix()
            self.calculate_similarity()
            self.save_model()
        
    def build_feature_matrix(self):
        """
        Build TF-IDF feature matrix from movie genres and tags
        """
        print("\n🔨 Building feature matrix...")
        
        # Create a copy to avoid modifying original
        movies_copy = self.movies.copy()
        
        # Create combined features (genres + tags)
        movies_copy['combined_features'] = movies_copy['genres']
        
        # Add tags to features
        if self.tags is not None and len(self.tags) > 0:
            # Group tags by movie
            tag_groups = self.tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
            movies_copy['combined_features'] = movies_copy['combined_features'] + ' ' + movies_copy['movieId'].map(tag_groups).fillna('')
        
        # Fill any NaN values
        movies_copy['combined_features'] = movies_copy['combined_features'].fillna('')
        
        # Store the combined features back
        self.movies['combined_features'] = movies_copy['combined_features']
        
        print(f"✅ Created features for {len(self.movies)} movies")
        return movies_copy
        
    def calculate_similarity(self):
        """
        Calculate cosine similarity between all movies using TF-IDF
        """
        print("\n📊 Calculating movie similarities...")
        
        # Make sure combined_features exists
        if 'combined_features' not in self.movies.columns:
            self.build_feature_matrix()
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies['combined_features'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create index mapping
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
        
        print(f"✅ Calculated similarities for {len(self.movies)} movies")
        print(f"✅ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"✅ Number of movie titles in index: {len(self.indices)}")
        
    def ensure_model_built(self):
        """
        Ensure the model is built before making recommendations
        """
        if self.indices is None or self.cosine_sim is None:
            print("⚠️ Model not built. Building now...")
            self.build_feature_matrix()
            self.calculate_similarity()
            self.save_model()
            return True
        return False
        
    def get_recommendations(self, title, n=10):
        """
        Get top N similar movies for a given movie title
        
        Args:
            title (str): Title of the movie
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N similar movies with similarity scores
        """
        # Ensure model is built
        self.ensure_model_built()
        
        # Check if movie exists
        if title not in self.indices:
            print(f"⚠️ Movie '{title}' not found in dataset!")
            # Try partial match
            matches = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
            if len(matches) > 0:
                title = matches.iloc[0]['title']
                print(f"✅ Using closest match: '{title}'")
            else:
                return pd.DataFrame()  # Return empty DataFrame
        
        # Get index of the movie
        idx = self.indices[title]
        
        # Get similarity scores for all movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N scores (skip first which is the movie itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Create results dataframe
        recommendations = self.movies.iloc[movie_indices][['title', 'genres']].copy()
        recommendations['similarity_score'] = [score[1] for score in sim_scores]
        
        return recommendations
    
    def search_movies(self, query):
        """
        Search for movies by title (partial match)
        
        Args:
            query (str): Search query
            
        Returns:
            DataFrame: Matching movies
        """
        matches = self.movies[self.movies['title'].str.contains(query, case=False, na=False)]
        return matches[['title', 'genres']].head(20)
    
    def get_movie_details(self, title):
        """
        Get detailed information about a movie
        
        Args:
            title (str): Movie title
            
        Returns:
            dict: Movie details
        """
        movie = self.movies[self.movies['title'] == title]
        if len(movie) == 0:
            return None
        return movie.iloc[0].to_dict()
    
    def save_model(self, path='models/content_based_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'movies': self.movies
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='models/content_based_model.pkl'):
        """
        Load a saved model from disk
        
        Args:
            path (str): Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.cosine_sim = model_data['cosine_sim']
        self.indices = model_data['indices']
        self.movies = model_data['movies']
        
        print(f"✅ Model loaded from {path}")
        print(f"   Movies in index: {len(self.indices)}")
        print(f"   Cosine sim shape: {self.cosine_sim.shape}")

def test_recommender():
    """
    Test the content-based recommender
    """
    print("\n" + "="*60)
    print("🎬 TESTING CONTENT-BASED RECOMMENDER")
    print("="*60)
    
    # Create recommender with force rebuild
    recommender = ContentBasedRecommender(force_rebuild=True)
    
    # Test recommendations for popular movies
    test_movies = ['Toy Story (1995)', 'Matrix, The (1999)', 'Forrest Gump (1994)', 'Titanic (1997)']
    
    for movie in test_movies:
        print(f"\n🎬 Recommendations for: {movie}")
        print("-" * 40)
        recommendations = recommender.get_recommendations(movie, n=5)
        if recommendations is not None and len(recommendations) > 0:
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"  {i}. {row['title']}")
                print(f"     Genres: {row['genres']}")
                print(f"     Similarity: {row['similarity_score']:.3f}")
        else:
            print("  No recommendations found")
    
    print("\n" + "="*60)
    print("✅ Content-Based Recommender test completed!")
    print("="*60)

if __name__ == "__main__":
    test_recommender()