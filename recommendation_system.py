import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_genres(genres):
    result = " ".join(genres.replace("-", "").split("|"))
    return result

def get_recommendations(title, top_k, df):
    data = df.loc[title, :]
    data = data.sort_values(ascending=False)
    return data[:top_k].to_frame(name="score")
    # return data

movies = pd.read_csv("movie_data/movies.csv", encoding="latin-1", sep="\t", usecols=["movie_id", "title", "genres"])
movies["genres"] = movies["genres"].apply(clean_genres)

tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(movies["genres"])
tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), columns=tf.get_feature_names(), index=movies["title"])

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, columns=movies["title"], index=movies["title"])

title = "Indian in the Cupboard, The (1995)"
top_k = 20
result = get_recommendations(title, top_k, cosine_sim_df)

