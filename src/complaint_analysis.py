# complaint_analysis.py

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

def download_nltk_resources():
    """Download NLTK resources needed for text processing."""
    print("Downloading NLTK resources...")
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    print("NLTK resources downloaded/checked.")

def initialize_nltk_components():
    """Initialize NLTK components for text processing."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def load_dataset(file_path):
    """Load the dataset from the specified file path."""
    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print("Dataset loaded successfully with UTF-8 encoding.")
    except UnicodeDecodeError:
        print("UTF-8 encoding failed. Trying 'latin1' encoding...")
        df = pd.read_csv(file_path, encoding='latin1')
        print("Dataset loaded successfully with 'latin1' encoding.")
    return df

def check_missing_values(df):
    """Check and return missing values in the dataset."""
    print("\n--- Missing Values Count ---")
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing %': missing_percentage})
    return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

def analyze_product_distribution(df):
    """Visualize the distribution of complaints by product."""
    print("\n--- Distribution of Complaints by Product ---")
    product_counts = df['Product'].value_counts()
    plt.figure(figsize=(12, 7))
    sns.barplot(x=product_counts.index, y=product_counts.values, palette='viridis')
    plt.title('Distribution of Complaints Across Products', fontsize=16)
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Number of Complaints', fontsize=12)
    plt.xticks(rotation=60, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def filter_data(df, products_to_keep):
    """Filter dataset to keep only specified products and remove empty narratives."""
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].astype(str).replace('nan', '')
    df_filtered = df[df['Product'].isin(products_to_keep)].copy()
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].str.strip().astype(bool)].copy()
    print(f"Filtered by products. New shape: {df_filtered.shape}")
    return df_filtered

def analyze_narrative_length(df):
    """Analyze and visualize the length of consumer complaint narratives."""
    df['narrative_length'] = df['Consumer complaint narrative'].apply(lambda x: len(x.split()))
    print("\nDescriptive statistics for narrative length (word count):")
    print(df['narrative_length'].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df['narrative_length'], bins=100, kde=True, color='skyblue', edgecolor='black')
    plt.title('Distribution of Consumer Complaint Narrative Lengths', fontsize=16)
    plt.xlabel('Number of Words in Narrative', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, df['narrative_length'].quantile(0.99) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.75)
    plt.show()

def clean_text(text, stop_words, lemmatizer):
    """Clean the text data."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b(xx+|xxx|xxxx)\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    filtered_words = [word for word in lemmatized_words if word not in stop_words and len(word) > 1]
    return ' '.join(filtered_words)

def apply_text_cleaning(df, stop_words, lemmatizer):
    """Clean the 'Consumer complaint narrative' column using stop words and lemmatization."""
    
    def process_text(text):
        return clean_text(text, stop_words, lemmatizer)
    
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(process_text)
    print("Text cleaning applied to 'Consumer complaint narrative' column.")


def save_cleaned_data(df, output_file_path):
    """Save the cleaned and filtered dataset to a CSV file."""
    final_df = df[['Product', 'cleaned_narrative']].copy()
    final_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"\nCleaned and filtered data saved successfully to: {output_file_path}")
    print(f"Shape of the saved data: {final_df.shape}")
    print("First 5 rows of the saved data (final output):")
    print(final_df.head())

def clean_up(df):
    """Clean up the DataFrame from memory."""
    del df
    print("In-memory DataFrames deleted.")