#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load dataset
df = pd.read_csv(r'C:\Users\nivi1\Downloads\archive (1)\IMDB_Dataset.csv')

# Preprocess text data
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

# Encode the sentiment column
y = y.apply(lambda x: 1 if x == 'positive' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

# Save the model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf, vec_file)


# In[ ]:


import tkinter as tk
from tkinter import messagebox
import pickle

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

# Function to predict sentiment
def predict_sentiment():
    review = text_input.get("1.0", tk.END).strip()
    if review:
        review_vectorized = tfidf.transform([review])
        prediction = model.predict(review_vectorized)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        sentiment_box.delete("1.0", tk.END)  # Clear the box before inserting the sentiment
        sentiment_box.insert(tk.END, sentiment)
    else:
        messagebox.showwarning("Input Error", "Please enter a movie review.")

# Create the main window
root = tk.Tk()
root.title("Movie Review Sentiment Analysis")

root.attributes("-fullscreen", True)

def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)
    return "break"

root.bind("<Escape>", exit_fullscreen)

# Create and place the label
label = tk.Label(root, text="Enter the movie review below:")
label.pack(pady=10)

# Create and place the text input widget
text_input = tk.Text(root, height=10, width=50)
text_input.pack(pady=20)

# Create and place the predict button
predict_button = tk.Button(root, text="Analyze Sentiment", command=predict_sentiment)
predict_button.pack(pady=10)

# Create and place the sentiment label box
sentiment_frame = tk.Frame(root)
sentiment_frame.pack(pady=10)

sentiment_label = tk.Label(sentiment_frame, text="Sentiment: ")
sentiment_label.pack(side=tk.LEFT)

sentiment_box = tk.Text(sentiment_frame, height=1, width=20)
sentiment_box.pack(side=tk.LEFT)

# Run the Tkinter event loop
root.mainloop()


# In[ ]:





# In[ ]:




