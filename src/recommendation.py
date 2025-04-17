import os
import re
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

GENRES = [
    'Unknown','Action','Adventure','Animation',"Children's",'Comedy',
    'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror',
    'Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
]

def load_data():
    """
    Load MovieLens 1M data:
    - ratings.dat: user_id, movie_id, rating, timestamp
    - movies.dat: movie_id, title, genres (pipe-separated)
    Returns two DataFrames: ratings and movies.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ml1m_dir = os.path.join(base_dir, 'data', 'ml-1m')
    ratings_fp = os.path.join(ml1m_dir, 'ratings.dat')
    movies_fp = os.path.join(ml1m_dir, 'movies.dat')

    ratings = pd.read_csv(
        ratings_fp,
        sep='::',
        engine='python',
        names=['user_id','movie_id','rating','timestamp']
    )

    movies = pd.read_csv(
        movies_fp,
        sep='::',
        engine='python',
        names=['movie_id','title','genres'],
        encoding='latin-1'
    )

    movies['genres'] = movies['genres'].str.split('|')

    movies['title'] = movies['title'].apply(clean_title)
    return ratings, movies


def clean_title(title):
    """
    Move 'A', 'An', 'The' at end of title to front.
    e.g. 'Close Shave, A (1995)' -> 'A Close Shave (1995)'
    """
    pattern = r'^(.*),\s*(A|An|The)\s*\((\d{4})\)$'
    m = re.match(pattern, title)
    if m:
        main, article, year = m.groups()
        return f"{article} {main} ({year})"
    return title


def train_model(ratings):
    """
    Train an SVD model using Surprise.
    1. Load ratings into Surprise dataset
    2. Perform cross-validation (RMSE, MAE)
    3. Fit model on entire dataset
    Returns the trained algorithm object.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)

    algo = SVD()
    cross_validate(algo, data, measures=['RMSE','MAE'], cv=3, verbose=False)

    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo


def get_recommendations(algo, movies, user_id, n=10):
    """
    Generate top-n recommendations for a user.
    1. Predict estimated rating for each movie
    2. Sort by estimated rating descending
    3. Return top-n movie_id and title
    """
    movie_ids = movies['movie_id'].unique()
    preds = [algo.predict(user_id, mid) for mid in movie_ids]
    preds.sort(key=lambda x: x.est, reverse=True)

    top_ids = [p.iid for p in preds[:n]]
    return movies[movies['movie_id'].isin(top_ids)][['movie_id','title']]


def get_user_ratings(movies, k=10):
    """
    Ask user to rate k random movies (cold-start).
    - Shows movie titles, collects ratings 1-5.
    - Assigns user_id=0, captures timestamp.
    """
    sample = movies[['movie_id','title']].sample(n=k).to_dict('records')
    user_ratings = []
    print("Please rate the following movies (1–5):")
    for rec in sample:
        while True:
            val = input(f" » '{rec['title']}': ")
            try:
                r = float(val)
                if 1 <= r <= 5:
                    break
            except:
                pass
            print("   Enter a numeric value between 1 and 5.")
        timestamp = pd.Timestamp.now().timestamp()
        user_ratings.append((0, rec['movie_id'], r, timestamp))
    return user_ratings


def ask_genres():
    """
    Prompt user to select favorite genres by number.
    Returns a set of chosen genre strings.
    """
    print("Select your favorite genres (comma-separated):")
    for i, g in enumerate(GENRES, 1):
        print(f"{i:2d}. {g}")
    choices = input("Numbers: ")
    idxs = {int(x) for x in choices.split(',') if x.strip().isdigit()}
    return {GENRES[i-1] for i in idxs if 1 <= i <= len(GENRES)}


def recommend_by_genre(movies, ratings, fav_genres, n=10):
    """
    Content-based recommendation by genre:
    1. Compute average rating per movie
    2. Filter movies matching favorite genres
    3. Sort by average rating and return top-n
    """
    avg = ratings.groupby('movie_id')['rating'].mean().rename('avg_rating')
    mask = movies['genres'].apply(lambda gl: bool(set(gl) & fav_genres))
    candidates = movies[mask].merge(avg, on='movie_id')
    return candidates.sort_values('avg_rating', ascending=False).head(n)[['movie_id','title','avg_rating']]


def main():
    """
    Main workflow:
    1. Load data
    2. Present menu: genre filter or cold-start rating
    3. Execute chosen recommendation method
    4. Display results
    """
    ratings, movies = load_data()

    print("\nWhich recommendation method would you like?")
    print(" 1) Filter by genres")
    print(" 2) Rate movies")
    choice = input("Select 1 or 2: ").strip()

    if choice == '1':
        fav = ask_genres()
        recs = recommend_by_genre(movies, ratings, fav, n=10)
        print("\nTop 10 based on your genres:")
        print(recs.to_string(index=False))

    elif choice == '2':
        new_r = get_user_ratings(movies, k=10)
        df_new = pd.DataFrame(new_r, columns=ratings.columns)
        ratings_ext = pd.concat([ratings, df_new], ignore_index=True)

        algo = train_model(ratings_ext)
        recs = get_recommendations(algo, movies, user_id=0, n=10)
        print("\nTop 10 recommendations for you:")
        print(recs.to_string(index=False))

    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == '__main__':
    main()