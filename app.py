import streamlit as st
import pytchat
import json
import time
import re
import threading
import signal
import os
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables from the .env file
load_dotenv()

# Get the Hugging Face token from the environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')

# Log in using the token
login(hf_token)

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Cache model and tokenizer to avoid reloading every time
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Custom keywords for sentiment adjustment
custom_positive_keywords = ['zabardast', 'lajawab', 'must', 'faadu']
custom_negative_keywords = ['bekaar', 'faltu', 'ganda']

# Function to adjust model prediction based on custom keywords
def adjust_sentiment_with_custom_keywords(message, model_prediction):
    message_lower = message.lower()
    if any(word in message_lower for word in custom_positive_keywords):
        return 'positive'
    elif any(word in message_lower for word in custom_negative_keywords):
        return 'negative'
    return model_prediction

# Function to predict sentiment using the pre-trained model
def classify_sentiment(message):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map predicted class index to sentiment label
    sentiment_labels = ['negative', 'neutral', 'positive']
    initial_sentiment = sentiment_labels[predicted_class]
    
    # Adjust sentiment based on custom keywords
    final_sentiment = adjust_sentiment_with_custom_keywords(message, initial_sentiment)
    return final_sentiment

# Function to process scraped data and find the user with the most negative comments
def process_scraped_data(file_path):
    # Load the scraped chat messages from the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    # Create a DataFrame for the messages
    df = pd.DataFrame(messages)

    # Apply sentiment classification to each message
    df['sentiment'] = df['message'].apply(classify_sentiment)

    # Calculate sentiment percentages
    total_messages = len(df)
    positive_count = df['sentiment'].value_counts().get('positive', 0)
    negative_count = df['sentiment'].value_counts().get('negative', 0)
    neutral_count = df['sentiment'].value_counts().get('neutral', 0)

    overall_positive_percentage = (positive_count / total_messages) * 100 if total_messages > 0 else 0
    overall_negative_percentage = (negative_count / total_messages) * 100 if total_messages > 0 else 0
    overall_neutral_percentage = (neutral_count / total_messages) * 100 if total_messages > 0 else 0

    # Find user with most negative comments
    negative_comments_df = df[df['sentiment'] == 'negative']  # Filter only negative comments
    if not negative_comments_df.empty:
        most_negative_user = negative_comments_df['author'].value_counts().idxmax()  # Find the user with the most negative comments
        negative_comment_count = negative_comments_df['author'].value_counts().max()  # Get the number of negative comments by that user
        st.write(f"\nUser with the most negative comments: {most_negative_user} ({negative_comment_count} negative comments)")
    else:
        st.write("No negative comments found.")

    # Display the results in the Streamlit app
    st.write("\nSentiment Analysis Results:")
    st.write(df[['author', 'message', 'sentiment']])
    st.write(f"\nOverall Positive Sentiment: {overall_positive_percentage:.2f}%")
    st.write(f"Overall Negative Sentiment: {overall_negative_percentage:.2f}%")
    st.write(f"Overall Neutral Sentiment: {overall_neutral_percentage:.2f}%")

# Function to extract video ID from YouTube URL
def get_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL or no video ID found.")

# Function to generate a unique filename based on the video ID and current timestamp
def generate_filename(video_id):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # e.g., 20230921-153045
    return f"live_chat_{video_id}_{timestamp}.json"

# Monkey patch to make signal handling a no-op in non-main threads
def patch_signal():
    if threading.current_thread() != threading.main_thread():
        original_signal = signal.signal
        signal.signal = lambda *args, **kwargs: None
        return original_signal
    return None

# Restore original signal handler
def restore_signal(original_signal):
    if original_signal:
        signal.signal = original_signal

# Function to run chat extraction in a background thread
def extract_chat(video_id, filename, duration=300):  # Default duration is 5 minutes
    original_signal = patch_signal()
    chat = pytchat.create(video_id=video_id)
    restore_signal(original_signal)

    start_time = time.time()
    messages = []

    while chat.is_alive() and (time.time() - start_time) < duration:
        for message in chat.get().sync_items():
            msg_dict = {
                'author': message.author.name,
                'message': message.message,
                'timestamp': message.timestamp,
            }
            messages.append(msg_dict)

        # Save messages to JSON file every 10 seconds
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)

        time.sleep(10)  # Adjust this delay as needed

    return messages  # Return messages for sentiment analysis

# Streamlit app
def main():
    st.title("YouTube Live Chat Extractor and Sentiment Analysis")

    # Input field for YouTube URL
    url = st.text_input("Enter YouTube live stream URL:")

    # Button to start scraping
    if st.button("Start Scraping"):
        if url:
            try:
                video_id = get_video_id(url)  # Extract video_id from the URL
                filename = generate_filename(video_id)  # Generate a unique filename
                
                # Start background thread to extract chat messages for 5 minutes
                st.success("Scraping started! It will run for 5 minutes.")
                messages = extract_chat(video_id, filename)  # Pass filename here

                st.success("Data scraped successfully!")
                
                # Process the scraped messages for sentiment analysis
                process_scraped_data(filename)  # Pass the filename for processing

            except ValueError as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()
