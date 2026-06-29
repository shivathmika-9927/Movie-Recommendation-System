"""
Collaborative Filtering Recommendation System for CineMatch AI
Uses SVD (Singular Value Decomposition) for recommendations
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from src.data_loader import MovieDataLoader
import pickle
import os

class CollaborativeRecommender:
    """
    Collaborative filtering recommender using SVD
    """
    
    def __init__(self):
        """Initialize the recommender with data"""
        self.loader = MovieDataLoader()
        self.movies, self.ratings, self.tags = self.loader.load_all()
        self.model = None
        self.trainset = None
        self.testset = None
        
    def prepare_data(self):
        """
        Prepare data for Surprise library
        """
        print("\n📊 Preparing data for collaborative filtering...")
        
        # Create reader with rating scale
        reader = Reader(rating_scale=(0.5, 5.0))
        
        # Load data into Surprise format
        data = Dataset.load_from_df(
            self.ratings[['userId', 'movieId', 'rating']],
            reader
        )
        
        print(f"✅ Data prepared: {len(self.ratings)} ratings from {self.loader.get_user_count()} users")
        return data
    
    def train_model(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Train SVD model on the data
        
        Args:
            n_factors (int): Number of latent factors
            n_epochs (int): Number of training epochs
            lr_all (float): Learning rate
            reg_all (float): Regularization parameter
        """
        print("\n🔨 Training SVD model...")
        print(f"   Factors: {n_factors}")
        print(f"   Epochs: {n_epochs}")
        print(f"   Learning Rate: {lr_all}")
        print(f"   Regularization: {reg_all}")
        
        # Prepare data
        data = self.prepare_data()
        
        # Split into train and test sets
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Initialize SVD model
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        
        # Train the model
        self.model.fit(self.trainset)
        print("✅ Model training completed!")
        
        # Evaluate on test set
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"📊 Model Performance:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        return rmse, mae
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if self.model is None:
            print("❌ Model not trained! Train the model first.")
            return None
        
        # Convert to Surprise format (they use string IDs)
        prediction = self.model.predict(str(user_id), str(movie_id))
        return prediction.est
    
    def get_user_recommendations(self, user_id, n=10):
        """
        Get top N movie recommendations for a user
        
        Args:
            user_id (int): User ID
            n (int): Number of recommendations
            
        Returns:
            DataFrame: Top N recommendations with predicted ratings
        """
        if self.model is None:
            print("❌ Model not trained! Train the model first.")
            return None
        
        # Get all movies the user hasn't rated
        user_rated = self.ratings[self.ratings['userId'] == user_id]['movieId'].tolist()
        all_movies = self.movies['movieId'].tolist()
        unrated_movies = [m for m in all_movies if m not in user_rated]
        
        # Predict ratings for all unrated movies
        predictions = []
        for movie_id in unrated_movies:
            pred = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, pred))
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_n = predictions[:n]
        
        # Create results dataframe
        results = []
        for movie_id, pred_rating in top_n:
            movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            results.append({
                'movieId': movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': pred_rating
            })
        
        return pd.DataFrame(results)
    
    def get_similar_users(self, user_id, n=5):
        """
        Find users with similar rating patterns
        
        Args:
            user_id (int): User ID
            n (int): Number of similar users
            
        Returns:
            DataFrame: Similar users with their ratings
        """
        # Get user's ratings
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        user_movies = set(user_ratings['movieId'].tolist())
        
        # Find users who rated similar movies
        all_users = self.ratings['userId'].unique()
        similarity_scores = []
        
        for other_user in all_users:
            if other_user == user_id:
                continue
            
            other_ratings = self.ratings[self.ratings['userId'] == other_user]
            other_movies = set(other_ratings['movieId'].tolist())
            
            # Calculate overlap
            common_movies = user_movies.intersection(other_movies)
            
            if len(common_movies) > 0:
                # Calculate average rating difference
                user_avg = user_ratings[user_ratings['movieId'].isin(common_movies)]['rating'].mean()
                other_avg = other_ratings[other_ratings['movieId'].isin(common_movies)]['rating'].mean()
                similarity = 1 - abs(user_avg - other_avg) / 5.0
                
                similarity_scores.append({
                    'userId': other_user,
                    'similarity': similarity,
                    'common_movies': len(common_movies),
                    'total_ratings': len(other_ratings)
                })
        
        # Sort by similarity
        similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        return pd.DataFrame(similarity_scores[:n])
    
    def save_model(self, path='models/svd_model.pkl'):
        """
        Save the trained model to disk
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            print("❌ No model to save!")
            return
        
        model_data = {
            'model': self.model,
            'trainset': self.trainset,
            'testset': self.testset,
            'movies': self.movies
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='models/svd_model.pkl'):
        """
        Load a saved model from disk
        
        Args:
            path (str): Path to the saved model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.trainset = model_data['trainset']
        self.testset = model_data['testset']
        self.movies = model_data['movies']
        
        print(f"✅ Model loaded from {path}")

def test_collaborative_recommender():
    """
    Test the collaborative recommender
    """
    print("\n" + "="*60)
    print("🎬 TESTING COLLABORATIVE FILTERING")
    print("="*60)
    
    # Create recommender
    recommender = CollaborativeRecommender()
    
    # Train model
    rmse, mae = recommender.train_model(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02
    )
    
    # Test predictions for a user
    test_user = 1
    print(f"\n👤 Getting recommendations for User {test_user}...")
    recommendations = recommender.get_user_recommendations(test_user, n=5)
    
    if recommendations is not None:
        print(f"\n🎬 Top 5 Recommendations for User {test_user}:")
        print("-" * 40)
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"  {i}. {row['title']}")
            print(f"     Genres: {row['genres']}")
            print(f"     Predicted Rating: {row['predicted_rating']:.2f}")
    
    # Save the model
    recommender.save_model()
    
    print("\n" + "="*60)
    print("✅ Collaborative Recommender test completed!")
    print("="*60)

if __name__ == "__main__":
    test_collaborative_recommender()