from sklearn.feature_extraction.text import TfidfVectorizer


def process_text(train_text, test_text, max_features=20):
    """
    Converts text data into TF-IDF features.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    X_train_text = tfidf.fit_transform(train_text)
    X_test_text = tfidf.transform(test_text)

    return X_train_text, X_test_text, tfidf
