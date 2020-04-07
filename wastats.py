import base64
import re
from io import BytesIO
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud


def startsWithDate(s):
    patterns = [
        '^(\d{4})/(0[1-9]|1[0-2]|[1-9])/([1-9]|0[1-9]|[1-2]\d|3[0-1]), ([0-9][0-9]):([0-9][0-9]) -',
        '^(1[0-2]|[1-9])/([1-9]|[1-2]\d|3[0-1])/(\d{2}), ([0-9][0-9]):([0-9][0-9]) -'
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False


def get_message(line):
    splitLine = line.split(' - ')
    message = ' '.join(splitLine[1:])
    splitMessage = message.split(': ')
    message = ' '.join(splitMessage[1:])
    return message


def process(file):
    text = ''
    with open(file, 'r', encoding='utf-8') as f:
        f.readline()
        message_buffer = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if startsWithDate(line):
                if len(message_buffer) > 0:
                    text = text + ' '.join(message_buffer) + '\n'
                message_buffer.clear()
                message = get_message(line)
                message_buffer.append(message)
            else:
                message_buffer.append(line)

    with open('stopwords.txt', 'r', encoding='utf-8') as sw:
        default_stopwords = set(nltk.corpus.stopwords.words('english'))
        stopwords = set(sw.read().splitlines())
        all_stopwords = default_stopwords | stopwords
    # Tokenize clean messages
    words = nltk.tokenize.word_tokenize(text)
    # Remove single-character tokens (mostly punctuation)
    words = [word for word in words if len(word) > 1]
    # Remove non-alpha
    words = [word for word in words if word.isalpha()]
    # Lowercase all words (default_stopwords are lowercase too)
    words = [word.lower() for word in words]
    # Remove stopwords
    words = [word for word in words if word not in all_stopwords]
    # Calculate word frequency
    freq = nltk.FreqDist(words)
    # Create dict of 500 most common words
    filter_words = dict([(m, n) for m, n in freq.most_common(500)])

    wordcloud = WordCloud(width=720, height=1280, max_font_size=100, max_words=1000, background_color="white")
    wordcloud.generate_from_frequencies(frequencies=filter_words)
    plt.figure(figsize=(9, 16), dpi=320, edgecolor='black')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data
