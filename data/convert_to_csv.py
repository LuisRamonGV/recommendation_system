import os
import pandas as pd

base_dir = os.path.dirname(__file__)
ml1m_dir = os.path.join(base_dir, 'ml-1m')

ratings = pd.read_csv(
    os.path.join(ml1m_dir, 'ratings.dat'),
    sep='::',
    engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)
ratings.to_csv(os.path.join(base_dir, 'ratings.csv'), index=False)
print(f"ratings.csv generado con {len(ratings)} filas y columnas {ratings.columns.tolist()}")

movies = pd.read_csv(
    os.path.join(ml1m_dir, 'movies.dat'),
    sep='::',
    engine='python',
    names=['movie_id', 'title', 'genres'],
    encoding='latin-1'
)
movies.to_csv(os.path.join(base_dir, 'movies.csv'), index=False)
print(f"movies.csv generado con {len(movies)} filas y columnas {movies.columns.tolist()}")
