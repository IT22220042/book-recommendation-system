from flask import Flask, request, jsonify, render_template, session, redirect,url_for
import pandas as pd
import random
import os
from flask_sqlalchemy import SQLAlchemy
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json

# create an object of flask
# this is built syntax we can't change this
app = Flask(__name__)

#request = when we create forms inside HTML, get and send data to backend

#render_template
#integrate frontend with backend
#also navigate from page to another page
#overall define - redirect from began to HTML pages.

# load files===========================================================================================================
trending_products = pd.read_csv("Models/trending_dataset.csv")
dataset = pd.read_csv("models/filtered_dataset.csv")

# database configuration---------------------------------------
app.secret_key = "AHRSRecommendo"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/book_reco_system"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    cart_history = db.Column(db.Text, nullable=True)
# Define your model class for the 'signup' table
# Update your Signin model to store a timestamp instead of the password
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    signin_time = db.Column(db.DateTime, default=datetime.utcnow)

# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text





def collaborative_filtering_recommendations(dataset, target_user_id, top_n=20):
    """
    Recommends items based on collaborative filtering using user-item interactions.

    Args:
    - dataset (pd.DataFrame): The dataset containing user-item interactions with columns ['User-ID', 'ISBN', 'Rating'].
    - target_user_id (int): The ID of the target user for whom to generate recommendations.
    - top_n (int): The number of top recommendations to return.

    Returns:
    - pd.DataFrame: A DataFrame with details of the top recommended items.
    """

    # Create the user-item matrix
    user_item_matrix = dataset.pivot_table(index='User-ID', columns='Title', values='Rating').fillna(0)

    # Calculate the user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)

    # Check if target_user_id exists in the matrix
    if target_user_id not in user_item_matrix.index:
        raise ValueError(f"User-ID {target_user_id} not found in dataset.")

    # Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)

    # Get the similarity scores for the target user
    user_similarities = user_similarity[target_user_index]

    # Sort the users by similarity in descending order (excluding the target user)
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    # Generate recommendations based on similar users
    recommended_items = []

    for user_index in similar_users_indices:
        # Get items rated by the similar user but not by the target user
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)

        # Extract the item Titles of recommended items
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user])

    # Ensure unique titles in recommendations
    recommended_items = list(set(recommended_items))

    # Get the details of recommended items (up to top_n unique titles)
    recommended_items_details = dataset[dataset['Title'].isin(recommended_items)][
        ['Title', 'Author', 'ImageURL', 'Rating' , 'Publication Year']].drop_duplicates(subset='Title').head(top_n)

    return recommended_items_details






def content_based_recommendations(dataset, item_name, top_n=10):
    """
    Recommends items based on content similarity using 'Title' or 'Author'.

    Args:
    - dataset (pd.DataFrame): The dataset containing items with columns ['Title', 'Author', 'ReviewCount', 'ImageURL', 'Rating'].
    - item_name (str): The name of the item (Title or Author) for which similar items are recommended.
    - top_n (int): The number of top recommendations to return.

    Returns:
    - pd.DataFrame: A DataFrame with details of the top recommended similar items.
    """

    # Clean the input book title or author name (lowercase, remove extra spaces, punctuation)
    def clean_text(text):
        text = text.lower().strip()
        return re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    cleaned_item_name = clean_text(item_name)

    # Clean titles and authors in the dataset
    dataset['cleaned_Title'] = dataset['Title'].apply(clean_text)
    dataset['cleaned_Author'] = dataset['Author'].apply(clean_text)

    # Check for an exact match for Title or Author
    exact_match = dataset[(dataset['cleaned_Title'] == cleaned_item_name) |
                          (dataset['cleaned_Author'] == cleaned_item_name)]

    if not exact_match.empty:
        item_name = exact_match.iloc[0]['Title']
    else:
        # If no exact match, fall back to partial match (str.contains) for both Title and Author
        item_matches = dataset[(dataset['cleaned_Title'].str.contains(cleaned_item_name)) |
                               (dataset['cleaned_Author'].str.contains(cleaned_item_name))]
        if item_matches.empty:
            print(f"Item '{item_name}' not found in the data.")
            return pd.DataFrame()  # Return empty DataFrame if no matches

        # Use the first matched item for content-based recommendations
        item_name = item_matches.iloc[0]['Title']

    # Combine 'Title' and 'Author' to create a feature for similarity comparison
    dataset['content'] = dataset['Title'].fillna('') + ' ' + dataset['Author'].fillna('')

    # Create a TF-IDF vectorizer for the combined 'Title' and 'Author'
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to the content (Title + Author)
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(dataset['content'])

    # Find the index of the matched item
    try:
        item_index = dataset[dataset['Title'] == item_name].index[0]
    except IndexError:
        print(f"Item '{item_name}' not found in the dataset.")
        return pd.DataFrame()  # Return empty DataFrame if not found

    # Get the cosine similarity scores for the item
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content[item_index], tfidf_matrix_content)

    # Sort similar items by similarity score in descending order
    similar_items = list(enumerate(cosine_similarities_content[0]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    recommended_items_details = []
    seen_titles = set()

    for index, score in similar_items[1:]:  # Skip the first one (itself)
        if len(recommended_items_details) >= top_n:
            break
        title = dataset.iloc[index]['Title']
        if title not in seen_titles:
            seen_titles.add(title)
            recommended_items_details.append(dataset.iloc[index][['Title', 'Author', 'ImageURL', 'Rating' , 'Publication Year', 'ISBN']])

    return pd.DataFrame(recommended_items_details).head(top_n)

#routes
@app.route("/")
def index():
    # Create a list of random image URLs for each product
    product_image_urls = trending_products['ImageURL'].tolist()
    price = [1000, 1500, 2000, 2450, 2750, 950, 800, 3000, 4000, 5000]
    error = None
    if request.method == 'POST':
        password = request.form['password']
        repassword = request.form['repassword']

        # Check if passwords match
        if password != repassword:
            error = "Passwords do not match. Please try again."

    # Render template with error message
    return render_template('index.html', error=error , trending_products=trending_products.head(20),truncate = truncate,
                           product_image_urls=product_image_urls[:10],
                           random_price = random.choice(price))


@app.route("/main") #name we give to href link
def main():
    return recommendations()

@app.route("/index") #name we give to href link
def indexReDirect():
    trending_products['ImageURL'] = trending_products['ImageURL']
    price = [1000, 1500, 2000, 2450, 2750, 950, 800, 3000, 4000, 5000]
    return render_template('index.html', trending_products=trending_products.head(20),truncate = truncate,
                           random_price = random.choice(price))


@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()


        price = [1000, 1500, 2000, 2450, 2750, 950, 800, 3000, 4000, 5000]
        return render_template('index.html', trending_products=trending_products.head(20), truncate=truncate,
                               random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    try:
        if 'username' in session:
            # Fetch the currently signed-in user's username
            current_username = session['username']

            # Fetch the user record from the database
            user = Signup.query.filter_by(username=current_username).first()

            if not user:
                return jsonify({"error": "User not found"}), 404

            # Get the item_id from the request data
            data = request.get_json()
            item_id = data.get('item_id')

            # Update the user's cart history (assuming it's stored as JSON)
            if user.cart_history:
                cart_history = json.loads(user.cart_history)
            else:
                cart_history = []

            cart_history.append(item_id)
            user.cart_history = json.dumps(cart_history)

            # Commit the changes to the database
            db.session.commit()

            return jsonify({"message": "Item added to cart successfully"}), 200
        else:
            return jsonify({"error": "User not signed in"}), 403
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

# Route for signin page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername'].strip()
        password = request.form['signinPassword'].strip()

        # Check if user exists in the database
        user = Signup.query.filter_by(username=username).first()

        if user and user.password == password:
            # Store username in session
            session['username'] = username
            session['logged_in'] = True

            # Log username and sign-in time in the Signin table (without storing the password)
            new_signin = Signin(username=username)
            db.session.add(new_signin)
            db.session.commit()

            # Create a list of random price for each product
            price = [1000, 1500, 2000, 2450, 2750, 950, 800, 3000, 4000, 5000]

            return render_template('index.html', trending_products=trending_products.head(20),
                                   truncate=truncate,
                                   random_price=random.choice(price), signup_message='User signed in successfully!')
        else:
            return render_template('index.html', trending_products=trending_products.head(0),
                                   truncate=truncate, signup_message='User sign-in failed!')

    # Ensure to return something in case of a GET request
    return render_template('main.html')  # Create a signin.html for the GET request if not exists


@app.route('/logout')
def logout():
    session.pop('username', None)
    session['logged_in'] = False
    return redirect('/')


@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    try:
        if 'username' in session:
            # Fetch the currently signed-in user's username
            current_username = session['username']
            user = Signup.query.filter_by(username=current_username).first()

            if not user:
                message = "User not found in the database."
                return render_template('main.html', message=message)

            target_user_id = user.id

            if request.method == 'POST':
                # Fetch the product name and number of recommendations from the form
                prod = request.form.get('prod')
                nbr = request.form.get('nbr')

                if not prod or not nbr:
                    message = "Please provide both product name and the number of recommendations."
                    return render_template('main.html', message=message)

                try:
                    nbr = int(nbr)
                except ValueError:
                    message = "The number of recommendations must be an integer."
                    return render_template('main.html', message=message)

                # Get content-based recommendations based on user input
                content_based_rec = content_based_recommendations(dataset, prod, top_n=nbr)

                if content_based_rec.empty:
                    message = f"No recommendations available for the product '{prod}'."
                    return render_template('main.html', message=message)
                else:
                    random_prices = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(content_based_rec))]
                    return render_template('main.html', content_based_rec=content_based_rec,
                                           random_prices=random_prices, truncate=truncate)

            else:
                # Handle GET request for collaborative filtering recommendations
                collaborative_rec = collaborative_filtering_recommendations(dataset, target_user_id)

                if collaborative_rec.empty:
                    message = "No collaborative recommendations available at this time."
                    return render_template('main.html', message=message)

                return render_template('main.html', truncate=truncate, collaborative_rec=collaborative_rec)

        else:
            # Handle the case when user is not logged in
            if request.method == 'POST':
                prod = request.form.get('prod')
                nbr = request.form.get('nbr')

                if not prod or not nbr:
                    message = "Please provide both product name and the number of recommendations."
                    return render_template('main.html', message=message)

                try:
                    nbr = int(nbr)
                except ValueError:
                    message = "The number of recommendations must be an integer."
                    return render_template('main.html', message=message)

                # Get content-based recommendations based on user input
                content_based_rec = content_based_recommendations(dataset, prod, top_n=nbr)

                if content_based_rec.empty:
                    message = f"No recommendations available for the product '{prod}'."
                    return render_template('main.html', message=message)
                else:
                    random_prices = [random.choice([40, 50, 60, 70, 100, 122, 106, 50, 30, 50]) for _ in range(len(content_based_rec))]
                    return render_template('main.html', content_based_rec=content_based_rec,
                                           random_prices=random_prices, truncate=truncate)

            # If GET request and not signed in, return main.html with a prompt or empty state
            return render_template('main.html', message="Please sign in to access personalized recommendations.")

    except Exception as e:
        print(f"Error: {e}")
        return render_template('main.html', message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)