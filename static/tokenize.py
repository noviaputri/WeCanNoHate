from nltk.tokenize import TweetTokenizer
def tokenize(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)