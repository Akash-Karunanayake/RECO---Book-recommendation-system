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

# Content-based recommendation function
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
        book_title = df.iloc[i]['bookTitle']
        if book_title not in unique_titles:
            unique_titles.add(book_title)
            filtered_sim_scores.append((i, score))
        # Stop if when have enough recommendations
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

    # Ensure 'bookID' is included
    return recommendations[['bookID', 'bookTitle', 'bookAuthor', 'bookCategory', 'Price', 'weighted_rating', 'image', 'score_content']]

# Collaborative filtering function
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
    recommendations = recommendations[['bookID', 'bookTitle', 'bookAuthor', 'bookCategory', 'Price', 'weighted_rating', 'image']].copy()
    recommendations['score_collaborative'] = recommendations['bookID'].map(dict(zip(top_book_ids, scores)))

    return recommendations

def hybrid_recommendations(content_recs, collaborative_recs, df, content_weight=0.5, collaborative_weight=0.5, top_n=10):
    # Merge recommendations on 'bookID' using an 'outer' join
    combined_recs = pd.merge(
        content_recs[['bookID', 'score_content']],
        collaborative_recs[['bookID', 'score_collaborative']],
        on='bookID',
        how='outer'  # Include all books from both recommendations
    )

    # Fill NaN scores with zeros
    combined_recs['score_content'] = combined_recs['score_content'].fillna(0)
    combined_recs['score_collaborative'] = combined_recs['score_collaborative'].fillna(0)

    # Normalize scores
    if combined_recs['score_content'].max() != 0:
        combined_recs['score_content_normalized'] = combined_recs['score_content'] / combined_recs['score_content'].max()
    else:
        combined_recs['score_content_normalized'] = 0

    if combined_recs['score_collaborative'].max() != 0:
        combined_recs['score_collaborative_normalized'] = combined_recs['score_collaborative'] / combined_recs['score_collaborative'].max()
    else:
        combined_recs['score_collaborative_normalized'] = 0

    # Compute the hybrid score
    combined_recs['score_hybrid'] = (
        content_weight * combined_recs['score_content_normalized'] +
        collaborative_weight * combined_recs['score_collaborative_normalized']
    )

    # Merge with the main DataFrame to get item attributes
    combined_recs = pd.merge(
        combined_recs,
        df[['bookID', 'bookTitle', 'bookAuthor', 'bookCategory', 'Price', 'weighted_rating', 'image']],
        on='bookID',
        how='left'
    )

    # Remove duplicates
    combined_recs = combined_recs.drop_duplicates(subset='bookID')

    # Sort and select top_n recommendations
    combined_recs = combined_recs.sort_values(by='score_hybrid', ascending=False)
    hybrid_top_n = combined_recs.head(top_n).reset_index(drop=True)

    # Rearrange columns
    columns_order = ['bookID', 'bookTitle', 'bookAuthor', 'bookCategory', 'Price',
                     'weighted_rating', 'image', 'score_hybrid']
    hybrid_top_n = hybrid_top_n[columns_order]

    return hybrid_top_n

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
        action = request.form.get('action')
        if action == 'select_book':
            if 'prod' in request.form:
                session['prod'] = request.form.get('prod')
                nbr = request.form.get('nbr')
                if nbr is None or not nbr.isdigit():
                    nbr = 5
                else:
                    nbr = int(nbr)
                session['nbr'] = nbr
        elif action == 'select_user':
            if 'userName' in request.form:
                session['userName'] = request.form.get('userName')
        elif action == 'get_recommendations':
            # Ensure both user and book are selected
            if 'userName' not in session or 'prod' not in session:
                message = "Please select both a user and a book before getting recommendations."
            else:
                # Proceed to generate recommendations
                pass  # handle this in the next steps

    # Retrieve selections from session
    user_name = session.get('userName')
    prod = session.get('prod')
    nbr = int(session.get('nbr', 5))

    # Initialize variables
    recently_rated_books = None
    hybrid_recs = None
    user_details = None
    message = None

    # Set selected_userName and selected_bookTitle based on session variables
    selected_userName = user_name
    selected_bookTitle = prod

    # Check if  should generate recommendations
    if request.method == 'POST' and request.form.get('action') == 'get_recommendations':
        if user_name and prod:
            # Both user and product selected, generate hybrid recommendations
            user_ids = df[df['userName'] == user_name]['userID'].unique()
            user_id = user_ids[0]  # Assuming userName is unique

            # Get user details
            user_details = df[df['userID'] == user_id][['userID', 'userName', 'rated_books_count']].drop_duplicates().iloc[0].to_dict()

            # Get recently rated books
            recently_rated_books = df[df['userID'] == user_id].sort_values(by='timestamp', ascending=False).head(5).to_dict(orient='records')

            # Get collaborative filtering recommendations
            collaborative_df = collaborative_filtering_recommendations(df, user_id, top_n=100)  # Get more items for merging

            # Get content-based recommendations
            content_based_df = contentBased_recommendations(prod, cosine_similarities_content, df, top_n=100)

            if not content_based_df.empty:
                # Generate hybrid recommendations
                hybrid_df = hybrid_recommendations(content_based_df, collaborative_df, df, top_n=nbr)
                hybrid_recs = hybrid_df.to_dict(orient='records')

                # Get selected book details
                selected_book = df[df['bookTitle'] == prod].drop_duplicates('bookID').iloc[0].to_dict()
            else:
                message = "No hybrid recommendations available for the selected book."
        else:
            message = "Please select both a user and a book to get hybrid recommendations."
    else:
        # Handle displaying selected user and book without generating recommendations
        if user_name:
            user_ids = df[df['userName'] == user_name]['userID'].unique()
            user_id = user_ids[0]  # Assuming userName is unique

            # Get user details
            user_details = df[df['userID'] == user_id][['userID', 'userName', 'rated_books_count']].drop_duplicates().iloc[0].to_dict()

            # Get recently rated books
            recently_rated_books = df[df['userID'] == user_id].sort_values(by='timestamp', ascending=False).head(5).to_dict(orient='records')

        if prod:
            selected_book = df[df['bookTitle'] == prod].drop_duplicates('bookID').iloc[0].to_dict()
        else:
            selected_book = None  # Ensure selected_book is None if no book is selected

    return render_template('main.html',
                           users=users.to_dict(orient='records'),
                           books=books,
                           selected_userName=selected_userName,
                           recently_rated_books=recently_rated_books,
                           hybrid_recs=hybrid_recs,
                           selected_book=selected_book,
                           selected_bookTitle=selected_bookTitle,
                           user_details=user_details,
                           message=message)



if __name__ == '__main__':
    app.run(debug=True)
