import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer

headline_one = '<h1>Nation\'s Top Pseudoscientists Harness High-Energy Quartz Crystal Capable Of Reversing Effects Of Being Gemini</h1>'

tweet = '@fat_meats, veggies are better than you think.'

ecg_text = 'An electrocardiogram is used to record the electrical conduction through a person\'s heart. The readings can be used to diagnose cardiac arrhythmias.'

brands = 'Salvation Army, YMCA, Boys & Girls Club of America'

survey_text = 'A YouGov study found that American\'s like Italian food more than any other country\'s cuisine.'

populated_island = 'Indonesia was founded in 1945. It contains the most populated island in the world, Java, with over 140 million people.'

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  
  pos_counts = Counter()

  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

#stopword removal
stop_words = set(stopwords.words('english')) 

tokenized_survey = word_tokenize(survey_text)

text_no_stops = [word for word in tokenized_survey if word not in stop_words]

#changing cases 
brands_lower = brands.lower()

brands_upper = brands.upper()

#noise_removal
headline_no_tag = re.sub(r'<.?h1>','',headline_one)

tweet_no_at = re.sub(r'@','',tweet)

#word and sentence tokenization
tokenized_by_word = word_tokenize(ecg_text)

tokenized_by_sentence = sent_tokenize(ecg_text)

#stemming_words
stemmer = PorterStemmer()

stemmed = [stemmer.stem(token) for token in tokenized_by_word]

#lemmatizer
lemmatizer = WordNetLemmatizer()

tokenized_string = word_tokenize(populated_island)

lemmatized_pos = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized_string]

print(headline_no_tag)
print(tweet_no_at)

print('Word Tokenization:')
print(tokenized_by_word)

print('Sentence Tokenization:')
print(tokenized_by_sentence)

print(f'Lowercased brands: {brands_lower}')
print(f'Uppercased brands: {brands_upper}')

print(f'Text without Stops: {text_no_stops}')

print('Stemmed Words:')
print(stemmed)

print(f'The lemmatized words are: {lemmatized_pos}')