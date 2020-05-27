import re
from nltk import RegexpParser, Tree
from pos_tagged_oz import pos_tagged_oz
from np_chunk_counter import np_chunk_counter

# Compiling and match

# characters are defined
character_1 = "Dorothy"
character_2 = "Henry"

# compile your regular expression here
regular_expression = re.compile("\w{7}")

# check for a match to character_1 here
result_1 = regular_expression.match(character_1)
print(result_1)

# store and print the matched text here
match_1 = result_1.group(0)
print(match_1)

# compile a regular expression to match a 7 character string of word characters and check for a match to character_2 here
result_2 = re.match("\w{7}",character_2)
print(result_2)

# searching and finding

# import L. Frank Baum's The Wonderful Wizard of Oz
oz_text = open("the_wizard_of_oz_text.txt",encoding='utf-8').read().lower()

# search oz_text for an occurrence of 'wizard' here
found_wizard = re.search("wizard",oz_text)
print(found_wizard)

# find all the occurrences of 'lion' in oz_text here
all_lions = re.findall("lion",oz_text)
print(all_lions)

# store and print the length of all_lions here
number_lions = len(all_lions)
print(number_lions)

#chunking_noun_phrases

# define noun-phrase chunk grammar here
chunk_grammar = "NP: {<DT>?<JJ>*<NN>}"

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# create a list to hold noun-phrase chunked sentences
np_chunked_oz = list()

# create a for loop through each pos-tagged sentence in pos_tagged_oz here
for pos_tagged_sentence in pos_tagged_oz:
  # chunk each sentence and append to np_chunked_oz here
  np_chunked_oz.append(chunk_parser.parse(pos_tagged_sentence))

# store and print the most common np-chunks here
most_common_np_chunks = np_chunk_counter(np_chunked_oz)
print(most_common_np_chunks)

#chunking_verb_phrases

# define verb phrase chunk grammar here
chunk_grammar = "VP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}"
#chunk_grammar = "VP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}"

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# create a list to hold verb-phrase chunked sentences
vp_chunked_oz = list()

# create for loop through each pos-tagged sentence in pos_tagged_oz here
for pos_tagged_sentence in pos_tagged_oz:
  # chunk each sentence and append to vp_chunked_oz here
  vp_chunked_oz.append(chunk_parser.parse(pos_tagged_sentence))
  
# store and print the most common vp-chunks here
most_common_vp_chunks = vp_chunk_counter(vp_chunked_oz)
print(most_common_vp_chunks)

# chunk_filtering

# define chunk grammar to chunk an entire sentence together
grammar = "Chunk: {<.*>+}"

# create RegexpParser object
parser = RegexpParser(grammar)

# chunk the pos-tagged sentence at index 230 in pos_tagged_oz
chunked_dancers = parser.parse(pos_tagged_oz[230])
print(chunked_dancers)

# define noun phrase chunk grammar using chunk filtering here
chunk_grammar = """NP: {<.*>+}
                       }<VB.?|IN>+{"""

# create RegexpParser object here
chunk_parser = RegexpParser(chunk_grammar)

# chunk and filter the pos-tagged sentence at index 230 in pos_tagged_oz here
filtered_dancers = chunk_parser.parse(pos_tagged_oz[230])
print(filtered_dancers)

# pretty_print the chunked and filtered sentence here
Tree.fromstring(str(filtered_dancers)).pretty_print()