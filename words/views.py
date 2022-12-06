from django.shortcuts import render
from kaznlp.morphology.taggers import TaggerHMM
from kaznlp.morphology.analyzers import AnalyzerDD
from kaznlp.tokenization.tokhmm import TokenizerHMM
import os
from .models import Text



def index2(request, text):

    obj = Text.objects.all()
    txts = []
    for i in obj:
        txts.append(i.text)
    return render(request, 'words/index.html', {'txts': txts})


def index(request, text):

    analyzer = AnalyzerDD()
    analyzer.load_model(os.path.join('kaznlp', 'morphology', 'mdl'))
    mdl = os.path.join('kaznlp', 'tokenization', 'tokhmm.mdl')

    tagger = TaggerHMM(lyzer=analyzer)
    tagger.load_model(os.path.join('kaznlp', 'morphology', 'mdl'))

    # text = u'Еңбек етсең ерінбей, тояды қарның тіленбей.'
    
    tokenizer = TokenizerHMM(model=mdl)
    result = dict()
    for sentence in tokenizer.tokenize(text):
        lower_sentence = map(lambda x: x.lower(), sentence)
        for i, a in enumerate(tagger.tag_sentence(lower_sentence)):
            result[sentence[i]]= {i.split('_')[0]: '_'.join(i.split('_')[1:]) for i in a.split()}

    return render(request, 'words/index.html', {'text': text, 'result': result})