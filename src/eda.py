"""
Exploratory Data Analysis for CineMatch AI
Visualizes and analyzes MovieLens dataset patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import MovieDataLoader

class MovieEDA:
    """
    Class to perform Exploratory Data Analysis on MovieLens dataset
    """
    
    def __init__(self):
        """Initialize EDA with data from loader"""
        self.loader = MovieDataLoader()
        self.movies, self.ratings, self.tags = self.loader.load_all()
        
    def basic_statistics(self):
        """Print basic statistics about the dataset"""
        print("\n" + "="*50)
        print("📊 BASIC DATASET STATISTICS")
        print("="*50)
        
        print(f"\n🎬 Movies: {len(self.movies)}")
        print(f"⭐ Ratings: {len(self.ratings)}")
        print(f"👤 Users: {self.loader.get_user_count()}")
        print(f"📝 Tags: {len(self.tags)}")
        
        # Rating statistics
        stats = self.loader.get_rating_stats()
        print(f"\n⭐ Rating Statistics:")
        print(f"   Average: {stats['mean']:.2f}")
        print(f"   Median: {stats['median']:.2f}")
        print(f"   Min: {stats['min']}")
        print(f"   Max: {stats['max']}")
        print(f"   Std Dev: {stats['std']:.2f}")
        
    def rating_distribution(self):
        """Analyze how ratings are distributed"""
        print("\n" + "="*50)
        print("⭐ RATING DISTRIBUTION")
        print("="*50)
        
        # Count ratings by value
        rating_counts = self.ratings['rating'].value_counts().sort_index()
        
        print("\nRating Distribution:")
        for rating, count in rating_counts.items():
            percentage = (count / len(self.ratings)) * 100
            print(f"  {rating:.1f} stars: {count:,} ratings ({percentage:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Distribution of Movie Ratings', fontsize=16)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of Ratings', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/rating_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved rating distribution chart to 'assets/rating_distribution.png'")
        plt.show()
        
    def genre_analysis(self):
        """Analyze movie genres popularity"""
        print("\n" + "="*50)
        print("🎬 GENRE ANALYSIS")
        print("="*50)
        
        # Split genres and count
        all_genres = []
        for genres in self.movies['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        
        print("\nTop 10 Most Common Genres:")
        for i, (genre, count) in enumerate(genre_counts.head(10).items(), 1):
            print(f"  {i}. {genre}: {count} movies")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        genre_counts.head(10).plot(kind='bar', color='coral', edgecolor='black')
        plt.title('Top 10 Most Common Movie Genres', fontsize=16)
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Number of Movies', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/genre_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved genre analysis chart to 'assets/genre_analysis.png'")
        plt.show()
        
        return genre_counts
    
    def user_activity(self):
        """Analyze user rating patterns"""
        print("\n" + "="*50)
        print("👤 USER ACTIVITY ANALYSIS")
        print("="*50)
        
        # Ratings per user
        user_ratings = self.ratings.groupby('userId')['rating'].count()
        
        print(f"\nUser Rating Statistics:")
        print(f"  Average ratings per user: {user_ratings.mean():.1f}")
        print(f"  Median ratings per user: {user_ratings.median():.0f}")
        print(f"  Min ratings by a user: {user_ratings.min()}")
        print(f"  Max ratings by a user: {user_ratings.max()}")
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        user_ratings.hist(bins=50, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Ratings per User', fontsize=14)
        plt.xlabel('Number of Ratings', fontsize=10)
        plt.ylabel('Number of Users', fontsize=10)
        plt.grid(alpha=0.3)
        
        # Boxplot using matplotlib directly
        plt.subplot(1, 2, 2)
        rating_values = user_ratings.values
        bp = plt.boxplot(rating_values, vert=True, patch_artist=True)
        
        # Customize boxplot
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)
            
        plt.title('Boxplot of Ratings per User', fontsize=14)
        plt.ylabel('Number of Ratings', fontsize=10)
        plt.xticks([1], ['All Users'])
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('assets/user_activity.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved user activity chart to 'assets/user_activity.png'")
        plt.show()
        
    def movie_popularity(self):
        """Find most popular movies"""
        print("\n" + "="*50)
        print("🎬 MOST POPULAR MOVIES")
        print("="*50)
        
        # Count ratings per movie
        movie_ratings = self.ratings.groupby('movieId')['rating'].count()
        movie_ratings = movie_ratings.sort_values(ascending=False)
        
        # Get top 10 movies
        top_movie_ids = movie_ratings.head(10).index
        
        print("\nTop 10 Most Rated Movies:")
        for i, movie_id in enumerate(top_movie_ids, 1):
            title = self.loader.get_movie_title(movie_id)
            rating_count = movie_ratings[movie_id]
            avg_rating = self.ratings[self.ratings['movieId'] == movie_id]['rating'].mean()
            print(f"  {i}. {title}")
            print(f"     {rating_count} ratings, Average: {avg_rating:.2f}")
        
        # Create visualization
        top_10 = movie_ratings.head(10)
        titles = [self.loader.get_movie_title(mid) for mid in top_10.index]
        
        plt.figure(figsize=(12, 6))
        plt.barh(titles, top_10.values, color='mediumpurple', edgecolor='black')
        plt.title('Top 10 Most Popular Movies (by number of ratings)', fontsize=16)
        plt.xlabel('Number of Ratings', fontsize=12)
        plt.ylabel('Movie Title', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/movie_popularity.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved movie popularity chart to 'assets/movie_popularity.png'")
        plt.show()
    
    def run_all_analysis(self):
        """Run all EDA analyses"""
        print("\n" + "🔥"*20)
        print("🚀 RUNNING COMPLETE EDA ANALYSIS")
        print("🔥"*20)
        
        self.basic_statistics()
        self.rating_distribution()
        self.genre_analysis()
        self.user_activity()
        self.movie_popularity()
        
        print("\n" + "="*50)
        print("✅ EDA COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\n📊 All visualizations saved to 'assets/' folder:")
        print("  - rating_distribution.png")
        print("  - genre_analysis.png")
        print("  - user_activity.png")
        print("  - movie_popularity.png")

if __name__ == "__main__":
    # Run EDA
    eda = MovieEDA()
    eda.run_all_analysis()