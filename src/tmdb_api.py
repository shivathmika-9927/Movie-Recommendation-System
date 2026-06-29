"""
TMDB API Integration for CineMatch AI
Fetches movie posters, details, and metadata with caching and error handling
"""

import requests
import os
from dotenv import load_dotenv
import streamlit as st
from functools import lru_cache
import time
import random

# Load environment variables
load_dotenv()

class TMDBAPI:
    """
    TMDB API client for fetching movie data with caching and error handling
    """
    
    def __init__(self):
        """Initialize TMDB API client"""
        self.api_key = os.getenv('TMDB_API_KEY')
        self.base_url = 'https://api.themoviedb.org/3'
        self.image_base_url = 'https://image.tmdb.org/t/p/'
        self.cache = {}
        
        if not self.api_key:
            st.warning("⚠️ TMDB API Key not found. Please add it to .env file")
            self.api_key = None
    
    def _make_request(self, url, params, max_retries=2):
        """
        Make a request with retry logic
        
        Args:
            url (str): URL to request
            params (dict): Request parameters
            max_retries (int): Number of retry attempts
            
        Returns:
            dict: Response JSON or None
        """
        if not self.api_key:
            return None
        
        # Add API key to params
        params['api_key'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                # Add a small delay between retries
                if attempt > 0:
                    time.sleep(1 * attempt)
                
                response = requests.get(url, params=params, timeout=3)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout on attempt {attempt + 1}")
                
            except requests.exceptions.ConnectionError:
                print(f"🔌 Connection error on attempt {attempt + 1}")
                # Wait longer on connection errors
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Request error: {e}")
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
        
        return None
    
    def search_movie(self, query):
        """
        Search for a movie by title with error handling
        
        Args:
            query (str): Movie title to search
            
        Returns:
            dict: Search results or None
        """
        if not self.api_key:
            return None
            
        url = f"{self.base_url}/search/movie"
        params = {
            'query': query,
            'language': 'en-US'
        }
        
        return self._make_request(url, params)
    
    def get_movie_details(self, movie_id):
        """
        Get detailed information about a movie
        
        Args:
            movie_id (int): TMDB movie ID
            
        Returns:
            dict: Movie details or None
        """
        if not self.api_key:
            return None
            
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'language': 'en-US',
            'append_to_response': 'credits'
        }
        
        return self._make_request(url, params)
    
    def get_movie_poster(self, poster_path, size='w342'):
        """
        Get movie poster URL
        
        Args:
            poster_path (str): Poster path from TMDB
            size (str): Image size
            
        Returns:
            str: Poster URL or None
        """
        if not poster_path:
            return None
        return f"{self.image_base_url}{size}{poster_path}"
    
    def search_and_get_poster(self, movie_title):
        """
        Search for a movie and get its poster URL with caching
        
        Args:
            movie_title (str): Movie title
            
        Returns:
            tuple: (poster_url, movie_details)
        """
        # Check cache first
        cache_key = f"poster_{movie_title}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return None, None
        
        # Search for the movie
        results = self.search_movie(movie_title)
        
        if not results or not results.get('results'):
            # Cache empty result
            self.cache[cache_key] = (None, None)
            return None, None
        
        # Get first result
        try:
            movie = results['results'][0]
            poster_url = self.get_movie_poster(movie.get('poster_path'))
            
            movie_details = {
                'tmdb_id': movie.get('id'),
                'title': movie.get('title'),
                'overview': movie.get('overview', ''),
                'release_date': movie.get('release_date', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'poster_url': poster_url
            }
            
            # Cache the result
            self.cache[cache_key] = (poster_url, movie_details)
            return poster_url, movie_details
            
        except Exception as e:
            print(f"❌ Error processing movie data: {e}")
            self.cache[cache_key] = (None, None)
            return None, None

# Function to display movie poster safely
def get_movie_poster_safe(tmdb, movie_title):
    """
    Safely get movie poster with error handling
    
    Args:
        tmdb: TMDBAPI instance
        movie_title (str): Movie title
        
    Returns:
        tuple: (poster_url, movie_details)
    """
    try:
        return tmdb.search_and_get_poster(movie_title)
    except Exception as e:
        print(f"⚠️ Could not fetch poster for '{movie_title}': {e}")
        return None, None