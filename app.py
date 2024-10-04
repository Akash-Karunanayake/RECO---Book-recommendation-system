from flask import Flask, request, render_template, session
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'abcd'

highest_rated = pd.read_csv('models/highest_rated.csv')
df=pd.read_csv('models/newwww.csv')

#content-based recommendation system
# Compute TF-IDF and cosine similarity
def compute_similarity(text_data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(text_data)
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    return cosine_similarities_content

cosine_similarities_content = compute_similarity(df['tags'])

stop_words = set(stopwords.words('english'))
# Text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def contentBased_recommendations(title, cosine_similarities, df, top_n=10):
    # Preprocess the input title
    input_title = preprocess_text(title)


    if not input_title:
        print(f"Title '{title}' not found in the dataset.")
        return pd.DataFrame()

    # Create a temporary column for processed titles
    df['processed_title'] = df['bookTitle'].apply(preprocess_text)

    # Find the index of the input title
    idx_list = df[df['processed_title'] == input_title].index.tolist()
    if not idx_list:
        print(f"Title '{title}' not found in the dataset.")
        df.drop('processed_title', axis=1, inplace=True)
        return pd.DataFrame()

    # use the first occurrence
    idx = idx_list[0]

    # Get similarity scores for all books
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Exclude the book itself and any other books with the same normalized title
    sim_scores = [score for score in sim_scores if score[0] not in idx_list]

    # Sort the books based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

   # Keep track of unique book titles
    unique_titles = set()
    filtered_sim_scores = []
    for i, score in sim_scores:
        book_title = df.iloc[i]['Title']
        if book_title not in unique_titles:
            unique_titles.add(book_title)
            filtered_sim_scores.append((i, score))
        # Stop if we have enough recommendations
        if len(filtered_sim_scores) >= top_n:
            break

    # Get the indices and scores of the top_n unique similar books
    book_indices = [i for i, _ in filtered_sim_scores]
    similarity_scores = [score for _, score in filtered_sim_scores]

    # Get the recommended books with their scores
    recommendations = df.iloc[book_indices].copy()
    recommendations['score_content'] = similarity_scores

    # Drop the temporary column
    df.drop('processed_title', axis=1, inplace=True)

    return recommendations


#Collaboratove algorithm

from surprise import SVD, Dataset, Reader

def collaborative_filtering_recommendations(df, target_user_id, top_n=10):
    # Prepare the data
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['userID', 'bookID', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Build the model
    algo = SVD()
    algo.fit(trainset)

    # Get a list of all book IDs
    all_book_ids = df['bookID'].unique()

    # Predict ratings for all books not rated by the user
    user_rated_books = df[df['userID'] == target_user_id]['bookID'].unique()
    books_to_predict = [iid for iid in all_book_ids if iid not in user_rated_books]

    predictions = [algo.predict(target_user_id, iid) for iid in books_to_predict]

    # Get top N recommendations
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]
    top_book_ids = [pred.iid for pred in top_predictions]
    scores = [pred.est for pred in top_predictions]

    # Get book details
    recommendations = df[df['bookID'].isin(top_book_ids)].drop_duplicates('bookID')
    recommendations = recommendations[['bookID','bookTitle','bookAuthor','bookCategory', 'Price', 'weighted_rating', 'image']].copy()
    recommendations['score_collaborative'] = scores

    return recommendations


# routes========================================================


# Route to index page
@app.route('/')
def index():
    books = highest_rated.to_dict(orient='records')
    return render_template('index.html', books=books)

@app.route('/main', methods=['POST', 'GET'])
def main():
    # Get all users and their names
    users = df[['userID', 'userName']].drop_duplicates()

    # Get the list of unique book titles
    books = df['bookTitle'].unique().tolist()

    return render_template('main.html',
                           users=users.to_dict(orient='records'),
                           books=books)

@app.route('/index')
def indexredirect():
    books = highest_rated.to_dict(orient='records')
    return render_template('index.html', books=books)

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    books = df['bookTitle'].unique().tolist()
    users = df[['userID', 'userName']].drop_duplicates()

    if request.method == 'POST':
        # Update session variables based on form data
        if 'userName' in request.form:
            session['userName'] = request.form.get('userName')
        if 'prod' in request.form:
            session['prod'] = request.form.get('prod')
            nbr = request.form.get('nbr')
            if nbr is None or not nbr.isdigit():
                nbr = 5
            else:
                nbr = int(nbr)
            session['nbr'] = nbr

    # Get selections from session
    user_name = session.get('userName')
    prod = session.get('prod')
    nbr = session.get('nbr', 5)

    # Initialize variables
    recently_rated_books = None
    collaborative_recs = None
    selected_userName = None
    content_based_rec = None
    selected_book = None
    selected_bookTitle = None
    user_details = None

    # Generate outputs based on selections
    if user_name:
        user_ids = df[df['userName'] == user_name]['userID'].unique()
        user_id = user_ids[0]  # Assuming userName is unique

        selected_userName = user_name

        # Get user details
        user_details = df[df['userID'] == user_id][['userID', 'userName', 'rated_books_count']].drop_duplicates().iloc[0].to_dict()

        # Get recently rated books
        recently_rated_books = df[df['userID'] == user_id].sort_values(by='timestamp', ascending=False).head(5).to_dict(orient='records')

        # Get collaborative filtering recommendations
        collaborative_recs = collaborative_filtering_recommendations(df, user_id, 5).to_dict(orient='records')

    if prod:
        nbr = int(nbr)
        content_based_df = contentBased_recommendations(prod, cosine_similarities_content, df, top_n=nbr)
        if not content_based_df.empty:
            content_based_rec = content_based_df.to_dict(orient='records')
            selected_book = df[df['bookTitle'] == prod].drop_duplicates('bookID').iloc[0].to_dict()
            selected_bookTitle = prod
        else:
            content_based_rec = None
            selected_book = None
            selected_bookTitle = None

    return render_template('main.html',
                           users=users.to_dict(orient='records'),
                           books=books,
                           selected_userName=selected_userName,
                           recently_rated_books=recently_rated_books,
                           collaborative_recs=collaborative_recs,
                           content_based_rec=content_based_rec,
                           selected_book=selected_book,
                           selected_bookTitle=selected_bookTitle,
                           user_details=user_details)

if __name__ == '__main__':
    app.run(debug=True)