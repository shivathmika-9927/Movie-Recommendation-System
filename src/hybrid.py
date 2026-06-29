"""
Hybrid Recommendation System for CineMatch AI
Combines Content-Based and Collaborative Filtering
"""

import pandas as pd
import numpy as np
from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
import pickle
import os
class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering
    """
    
    def __init__(self, content_weight=0.5, collab_weight=0.5):
        """
        Initialize hybrid recommender
        
        Args:
            content_weight (float): Weight for content-based recommendations
            collab_weight (float): Weight for collaborative recommendations
        """
        self.loader = MovieDataLoader()
        self.movies, self.ratings, self.tags = self.loader.load_all()
        
        # Initialize recommenders
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeRecommender()
        
        # Set weights
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        
        # Load or build models
        self._load_or_build_models()
        
    def _load_or_build_models(self):
        """
        Load existing models or build new ones
        """
        print("\n🔧 Loading recommendation models...")
        
        # Try to load content-based model
        try:
            self.content_recommender.load_model('models/content_based_model.pkl')
        except:
            print("⚠️ Content-based model not found. Building new one...")
            self.content_recommender.build_feature_matrix()
            self.content_recommender.calculate_similarity()
            self.content_recommender.save_model()
        
        # Try to load collaborative model
        try:
            self.collab_recommender.load_model('models/svd_model.pkl')
        except:
            print("⚠️ Collaborative model not found. Building new one...")
            self.collab_recommender.train_model()
            self.collab_recommender.save_model()
        
        print("✅ All models loaded successfully!")
    
    def _find_movie_title(self, title_query):
        """
        Find the exact movie title from a partial match
        
        Args:
            title_query (str): Partial or full movie title
            
        Returns:
            str: Exact movie title or None if not found
        """
        # Try exact match first
        if title_query in self.content_recommender.indices:
            return title_query
        
        # Try partial match (case insensitive)
        matches = self.movies[self.movies['title'].str.contains(title_query, case=False, na=False)]
        if len(matches) > 0:
            return matches.iloc[0]['title']
        
        return None
    
    def get_recommendations(self, movie_title=None, user_id=None, n=10):
        """
        Get hybrid recommendations
        
        Args:
            movie_title (str): Movie title for content-based recommendations
            user_id (int): User ID for collaborative recommendations
            n (int): Number of recommendations
            
        Returns:
            DataFrame: Hybrid recommendations
        """
        recommendations = []
        
        # If movie_title is provided, find the exact match
        exact_title = None
        if movie_title:
            exact_title = self._find_movie_title(movie_title)
            if exact_title is None:
                print(f"⚠️ Movie '{movie_title}' not found. Using collaborative only.")
        
        # Get content-based recommendations
        if exact_title:
            content_recs = self.content_recommender.get_recommendations(exact_title, n=n*2)
            if content_recs is not None:
                for _, row in content_recs.iterrows():
                    recommendations.append({
                        'title': row['title'],
                        'genres': row['genres'],
                        'content_score': row['similarity_score'],
                        'collab_score': 0,
                        'hybrid_score': row['similarity_score'] * self.content_weight
                    })
        
        # Get collaborative recommendations
        if user_id:
            collab_recs = self.collab_recommender.get_user_recommendations(user_id, n=n*2)
            if collab_recs is not None:
                for _, row in collab_recs.iterrows():
                    # Check if movie already in recommendations
                    existing = next((r for r in recommendations if r['title'] == row['title']), None)
                    if existing:
                        existing['collab_score'] = row['predicted_rating'] / 5.0
                        existing['hybrid_score'] = (existing['content_score'] * self.content_weight + 
                                                   existing['collab_score'] * self.collab_weight)
                    else:
                        recommendations.append({
                            'title': row['title'],
                            'genres': row['genres'],
                            'content_score': 0,
                            'collab_score': row['predicted_rating'] / 5.0,
                            'hybrid_score': row['predicted_rating'] / 5.0 * self.collab_weight
                        })
        
        # If no recommendations yet, use collaborative
        if not recommendations and user_id:
            collab_recs = self.collab_recommender.get_user_recommendations(user_id, n=n)
            if collab_recs is not None:
                for _, row in collab_recs.iterrows():
                    recommendations.append({
                        'title': row['title'],
                        'genres': row['genres'],
                        'content_score': 0,
                        'collab_score': row['predicted_rating'] / 5.0,
                        'hybrid_score': row['predicted_rating'] / 5.0
                    })
        
        # Sort by hybrid score
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Convert to DataFrame
        if len(recommendations) > 0:
            result_df = pd.DataFrame(recommendations[:n])
            # Add movie IDs
            result_df['movieId'] = result_df['title'].apply(
                lambda x: self.movies[self.movies['title'] == x]['movieId'].values[0] 
                if len(self.movies[self.movies['title'] == x]) > 0 else None
            )
            return result_df
        
        return pd.DataFrame()
    
    def get_weighted_recommendations(self, movie_title=None, user_id=None, n=10):
        """
        Get weighted hybrid recommendations with more control
        
        Args:
            movie_title (str): Movie title for content-based
            user_id (int): User ID for collaborative
            n (int): Number of recommendations
            
        Returns:
            DataFrame: Weighted recommendations
        """
        recommendations_dict = {}
        
        # Find exact movie title if provided
        exact_title = None
        if movie_title:
            exact_title = self._find_movie_title(movie_title)
        
        # Get content-based recommendations
        if exact_title:
            content_recs = self.content_recommender.get_recommendations(exact_title, n=50)
            if content_recs is not None:
                for _, row in content_recs.iterrows():
                    title = row['title']
                    recommendations_dict[title] = {
                        'title': title,
                        'genres': row['genres'],
                        'content_score': row['similarity_score'] * 100,
                        'collab_score': 0,
                        'score': row['similarity_score'] * 100 * self.content_weight
                    }
        
        # Get collaborative recommendations
        if user_id:
            collab_recs = self.collab_recommender.get_user_recommendations(user_id, n=50)
            if collab_recs is not None:
                for _, row in collab_recs.iterrows():
                    title = row['title']
                    collab_score = row['predicted_rating'] / 5.0 * 100
                    if title in recommendations_dict:
                        recommendations_dict[title]['collab_score'] = collab_score
                        recommendations_dict[title]['score'] = (
                            recommendations_dict[title]['content_score'] * self.content_weight +
                            collab_score * self.collab_weight
                        )
                    else:
                        recommendations_dict[title] = {
                            'title': title,
                            'genres': row['genres'],
                            'content_score': 0,
                            'collab_score': collab_score,
                            'score': collab_score * self.collab_weight
                        }
        
        # Convert to DataFrame and sort
        results = pd.DataFrame(list(recommendations_dict.values()))
        if len(results) > 0:
            results = results.sort_values('score', ascending=False).head(n)
            
            # Add movie IDs
            results['movieId'] = results['title'].apply(
                lambda x: self.movies[self.movies['title'] == x]['movieId'].values[0] 
                if len(self.movies[self.movies['title'] == x]) > 0 else None
            )
        
        return results
    
    def get_diverse_recommendations(self, movie_title=None, user_id=None, n=10):
        """
        Get diverse recommendations (different genres)
        
        Args:
            movie_title (str): Movie title for content-based
            user_id (int): User ID for collaborative
            n (int): Number of recommendations
            
        Returns:
            DataFrame: Diverse recommendations
        """
        # Get all recommendations
        all_recs = self.get_recommendations(movie_title, user_id, n=50)
        
        if len(all_recs) == 0:
            return pd.DataFrame()
        
        # Group by genre and select top from each
        diverse_recs = []
        seen_genres = set()
        
        for _, row in all_recs.iterrows():
            genres = row['genres'].split('|')
            # Check if any genre is new
            if not seen_genres.intersection(set(genres)):
                diverse_recs.append(row.to_dict())  # Convert Series to dict
                seen_genres.update(genres)
            
            if len(diverse_recs) >= n:
                break
        
        # If we don't have enough, fill with remaining
        if len(diverse_recs) < n:
            remaining = all_recs[~all_recs['title'].isin([r['title'] for r in diverse_recs])]
            for _, row in remaining.head(n - len(diverse_recs)).iterrows():
                diverse_recs.append(row.to_dict())
        
        return pd.DataFrame(diverse_recs)

def test_hybrid_recommender():
    """
    Test the hybrid recommender
    """
    print("\n" + "="*60)
    print("🎬 TESTING HYBRID RECOMMENDER")
    print("="*60)
    
    # Create hybrid recommender
    hybrid = HybridRecommender(content_weight=0.4, collab_weight=0.6)
    
    # Test with a movie and user
    # Use exact title format from dataset
    test_movie = "Matrix, The (1999)"  # Correct format from MovieLens
    test_user = 1
    
    print(f"\n🎯 Testing with Movie: {test_movie} and User: {test_user}")
    print("-" * 40)
    
    # Get hybrid recommendations
    recommendations = hybrid.get_recommendations(
        movie_title=test_movie,
        user_id=test_user,
        n=10
    )
    
    if len(recommendations) > 0:
        print(f"\n🎬 Top 10 Hybrid Recommendations:")
        print("-" * 60)
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"  {i}. {row['title']}")
            print(f"     Genres: {row['genres']}")
            print(f"     Score: {row['hybrid_score']:.3f}")
            print()
    else:
        print("❌ No recommendations found!")
    
    # Test diverse recommendations
    print("\n🎯 Testing Diverse Recommendations:")
    print("-" * 40)
    diverse = hybrid.get_diverse_recommendations(
        movie_title=test_movie,
        user_id=test_user,
        n=5
    )
    
    if len(diverse) > 0:
        print(f"\n🎬 5 Diverse Recommendations:")
        print("-" * 60)
        for i, (_, row) in enumerate(diverse.iterrows(), 1):
            print(f"  {i}. {row['title']}")
            print(f"     Genres: {row['genres']}")
            print(f"     Score: {row['hybrid_score']:.3f}")
            print()
    else:
        print("❌ No diverse recommendations found!")
    
    print("\n" + "="*60)
    print("✅ Hybrid Recommender test completed!")
    print("="*60)

if __name__ == "__main__":
    test_hybrid_recommender()