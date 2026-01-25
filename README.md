# 🎧 Spotify Playlist ML App

A full-stack web application that automatically creates **genre-based playlists** from a user’s Spotify “Liked Songs.”  
The app connects to the **Spotify Web API** using `Spotipy`, extracts audio features, classifies songs by genre using **machine learning models**, and generates curated playlists — all from a clean web interface.

---

##  Features

-  **Spotify Authentication:** Securely connects with the user’s Spotify account using OAuth.
-  **Audio Feature Extraction:** Uses `Librosa` and Spotify’s API to gather MFCCs, chroma, tempo, and other features.
-  **Machine Learning Classification:** Classifies songs by genre using trained **CNN** and **LightGBM** models.
-  **Data Pipeline:** Handles preprocessing, caching, and feature storage using **SQLite** and **Pandas**.
-  **Interactive Web Interface:** Built with **Flask**, **HTML/CSS/JS** for playlist visualization and user control.
-  **Persistent Caching:** Reduces redundant Spotify API calls by caching preprocessed data.
-  **Modular Design:** Cleanly separated modules for ML, backend, and frontend components.



##  Tech Stack

**Languages & Frameworks:**  
- Python (Flask, Spotipy, TensorFlow, Scikit-learn, LightGBM, Librosa)  
- HTML, CSS, JavaScript  

**Tools & Libraries:**  
- Pandas, NumPy, Matplotlib, Seaborn  
- SQLite for caching and persistence  
- GitHub for version control  


##  Dataset

The models were trained using a combination of public audio genre datasets and Spotify track features, including:

- GTZAN Genre Dataset – 1,000 audio tracks across 10 genres (used for CNN training and MFCC/chroma extraction).

- Spotify Audio Features – Retrieved via the Spotify API (tempo, energy, valence, danceability, key, mode, loudness, etc.).

- Custom Extracted Features – MFCCs, spectral contrast, chroma, mel-scaled features generated using Librosa for each track.

## ⚙️ Installation & Setup

### 1. Clone the repository
```
git clone https://github.com/yourusername/spotify-ml-playlist.git
```
```
cd spotify-ml-playlist
```
### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up Spotify API credentials

Create a new Spotify Developer app at https://developer.spotify.com/dashboard
Then, create a .env file in the root directory:

```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:5000/callback
```
### 5. Run the app

```
python app.py
```

