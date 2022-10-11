from kaznlp.morphology.taggers import TaggerHMM
from kaznlp.morphology.analyzers import AnalyzerDD
from kaznlp.tokenization.tokhmm import TokenizerHMM
import os

# create a morphological analyzer instance
analyzer = AnalyzerDD()
# load the model directory located in the morphology directory
analyzer.load_model(os.path.join('kaznlp', 'morphology', 'mdl'))
mdl = os.path.join('kaznlp', 'tokenization', 'tokhmm.mdl')

tagger = TaggerHMM(lyzer=analyzer)
# same model directory is used to train the tagger
tagger.load_model(os.path.join('kaznlp', 'morphology', 'mdl'))

txt = u'Еңбек Путин етсең ерінбей, тояды қарның тіленбей алмасын.'

# to tag a text we need to split it into sentences
# and feed tokenized sentences to the *tag_sentence* method
tokenizer = TokenizerHMM(model=mdl)
for sentence in tokenizer.tokenize(txt):
    print(f'input sentence:\n{sentence}\n')
    print('tagged sentence:')
    #the sentence must be lower-cased before tagging
    lower_sentence = map(lambda x: x.lower(), sentence)
    for i, a in enumerate(tagger.tag_sentence(lower_sentence)):
        #output each word with the most probable analysis
        print(sentence[i])
        print({i.split('_')[0]: '_'.join(i.split('_')[1:]) for i in a.split()})

