# 🎬 CineMatch AI - Movie Recommendation System

## 📌 Overview

CineMatch AI is a professional **Movie Recommendation System** that combines **Content-Based Filtering**, **Collaborative Filtering (SVD)**, and **Hybrid Recommendations** to suggest movies users are likely to enjoy. The application is built using **Python**, **Streamlit**, **Scikit-learn**, and the **TMDB API** to provide an interactive, Netflix-style user experience.

---

## ✨ Features

* 🔍 Search movies instantly
* 🎯 Content-Based movie recommendations
* 👥 Collaborative Filtering using SVD
* 🤝 Hybrid recommendation engine
* 🎬 TMDB API integration for posters and movie details
* 🎨 Netflix-inspired user interface
* 📊 Interactive analytics dashboard

---

## 🛠️ Tech Stack

| Category             | Technologies                 |
| :------------------- | :--------------------------- |
| **Language**         | Python 3.8+                  |
| **Data Processing**  | Pandas, NumPy                |
| **Machine Learning** | Scikit-learn, Surprise (SVD) |
| **API**              | TMDB API                     |
| **Frontend**         | Streamlit                    |
| **Visualization**    | Matplotlib                   |
| **Version Control**  | Git & GitHub                 |

---

## 📁 Project Structure

```text
Movie-Recommendation-System/
├── data/
│   └── ml-latest-small/
│       ├── movies.csv
│       ├── ratings.csv
│       └── tags.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── content_based.py
│   ├── collaborative.py
│   ├── hybrid.py
│   └── tmdb_api.py
├── models/
│   ├── content_model.pkl
│   └── svd_model.pkl
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 Installation

### Prerequisites

* Python 3.8 or above
* Free TMDB API Key

### 1. Clone the Repository

```bash
git clone https://github.com/shivathmika-9927/Movie-Recommendation-System.git

cd Movie-Recommendation-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your TMDB API Key

Create a file named `.env`

```env
TMDB_API_KEY=your_tmdb_api_key
```

### 5. Download the Dataset

Download the **MovieLens Latest Small** dataset and extract it into:

```text
data/ml-latest-small/
```

The folder should contain:

```text
movies.csv
ratings.csv
tags.csv
```

### 6. Run the Application

```bash
streamlit run app.py
```

---

## 🎯 Recommendation Techniques

### Content-Based Filtering

* Uses TF-IDF Vectorization
* Calculates Cosine Similarity
* Finds movies similar to the selected movie

### Collaborative Filtering

* Uses Singular Value Decomposition (SVD)
* Learns user preferences
* Predicts ratings for unseen movies

### Hybrid Recommendation System

* Combines Content-Based and Collaborative Filtering
* Produces more accurate recommendations
* Handles cold-start scenarios effectively

---

## 📊 Dataset Statistics

| Metric         | Value   |
| :------------- | :------ |
| Movies         | 9,742   |
| Ratings        | 100,836 |
| Users          | 610     |
| Average Rating | 3.5     |

---

## 🔑 TMDB API Integration

The application uses the TMDB API to retrieve:

* 🎬 Movie posters
* 📝 Movie descriptions
* ⭐ Ratings
* 📅 Release dates
* 🔍 Search results

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes.

```bash
git commit -m "Add YourFeature"
```

4. Push your branch.

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

* MovieLens Dataset
* TMDB API
* Streamlit
* Scikit-learn
* Surprise Library

---

## 📞 Contact

**GitHub:** https://github.com/shivathmika-9927

**Repository:** https://github.com/shivathmika-9927/Movie-Recommendation-System

---

### ⭐ If you found this project helpful, please consider giving it a star!
