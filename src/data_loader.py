"""
Data Loader Module for CineMatch AI
Handles loading and basic preprocessing of MovieLens dataset
"""

import pandas as pd
import os

class MovieDataLoader:
    """
    A class to load and manage MovieLens dataset
    """
    
    def __init__(self, data_path='data/ml-latest-small/'):
        """
        Initialize the data loader with path to data
        
        Args:
            data_path (str): Path to the MovieLens dataset folder
        """
        self.data_path = data_path
        self.movies = None
        self.ratings = None
        self.tags = None
        
    def load_all(self):
        """
        Load all datasets (movies, ratings, tags)
        
        Returns:
            tuple: (movies_df, ratings_df, tags_df)
        """
        print("📊 Loading MovieLens dataset...")
        
        # Load movies
        self.movies = pd.read_csv(f'{self.data_path}movies.csv')
        print(f"✅ Loaded {len(self.movies)} movies")
        
        # Load ratings
        self.ratings = pd.read_csv(f'{self.data_path}ratings.csv')
        print(f"✅ Loaded {len(self.ratings)} ratings")
        
        # Load tags
        self.tags = pd.read_csv(f'{self.data_path}tags.csv')
        print(f"✅ Loaded {len(self.tags)} tags")
        
        return self.movies, self.ratings, self.tags
    
    def get_movie_info(self, movie_id):
        """
        Get information about a specific movie
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            dict: Movie information
        """
        movie = self.movies[self.movies['movieId'] == movie_id]
        if len(movie) == 0:
            return None
        return movie.iloc[0].to_dict()
    
    def get_movie_ratings(self, movie_id):
        """
        Get all ratings for a specific movie
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            DataFrame: All ratings for the movie
        """
        return self.ratings[self.ratings['movieId'] == movie_id]
    
    def get_user_ratings(self, user_id):
        """
        Get all ratings by a specific user
        
        Args:
            user_id (int): User ID
            
        Returns:
            DataFrame: All ratings by the user
        """
        return self.ratings[self.ratings['userId'] == user_id]
    
    def get_movie_genres(self, movie_id):
        """
        Get genres of a specific movie
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            list: List of genres
        """
        movie = self.movies[self.movies['movieId'] == movie_id]
        if len(movie) == 0:
            return []
        genres_str = movie.iloc[0]['genres']
        return genres_str.split('|')
    
    def get_movie_title(self, movie_id):
        """
        Get title of a specific movie
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            str: Movie title
        """
        movie = self.movies[self.movies['movieId'] == movie_id]
        if len(movie) == 0:
            return None
        return movie.iloc[0]['title']
    
    def get_movie_count(self):
        """Return total number of movies"""
        return len(self.movies)
    
    def get_user_count(self):
        """Return total number of unique users"""
        return self.ratings['userId'].nunique()
    
    def get_rating_stats(self):
        """
        Get statistics about ratings
        
        Returns:
            dict: Rating statistics
        """
        return {
            'mean': self.ratings['rating'].mean(),
            'median': self.ratings['rating'].median(),
            'min': self.ratings['rating'].min(),
            'max': self.ratings['rating'].max(),
            'std': self.ratings['rating'].std()
        }

# Quick test function
def test_data_loader():
    """
    Test function to verify data loader works
    """
    print("🧪 Testing Data Loader...")
    
    # Create loader instance
    loader = MovieDataLoader()
    
    # Load data
    movies, ratings, tags = loader.load_all()
    
    # Print basic info
    print("\n📊 Dataset Statistics:")
    print(f"Total Movies: {loader.get_movie_count()}")
    print(f"Total Users: {loader.get_user_count()}")
    print(f"Total Ratings: {len(ratings)}")
    
    # Print rating statistics
    stats = loader.get_rating_stats()
    print(f"\n⭐ Rating Statistics:")
    print(f"Average Rating: {stats['mean']:.2f}")
    print(f"Median Rating: {stats['median']:.2f}")
    print(f"Min Rating: {stats['min']}")
    print(f"Max Rating: {stats['max']}")
    
    # Show sample movies
    print("\n🎬 Sample Movies:")
    print(movies.head(3))
    
    # Show sample ratings
    print("\n⭐ Sample Ratings:")
    print(ratings.head(3))
    
    print("\n✅ Data Loader test completed successfully!")

if __name__ == "__main__":
    test_data_loader()