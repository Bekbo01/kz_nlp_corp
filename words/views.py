from django.shortcuts import render
from kaznlp.morphology.taggers import TaggerHMM
from kaznlp.morphology.analyzers import AnalyzerDD
from kaznlp.tokenization.tokhmm import TokenizerHMM
import os
from .models import Text
from django.views.generic import ListView, UpdateView
from .forms import TextForm


class TextsListView(ListView):
    model = Text
    template_name = 'texts.html'
    context_object_name = 'texts'
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super(TextsListView, self).get_context_data(**kwargs)
        texts_list = []
        for text in context['texts']:
            # if not self.request.user.has_perm('corpus.can_view_restricted_poems') and poem.is_restricted:
            #     continue
            text.name = text.get_name()
            texts_list.append(text)
        return context


class TextView(UpdateView):
    model = Text
    template_name = 'text.html'
    context_object_name = 'text'
    form_class = TextForm

    def get_context_data(self, **kwargs):
        context = super(TextView, self).get_context_data(**kwargs)
        poem = self.get_object()
        max_pk = Text.objects.latest('pk').pk

        next_pk = poem.pk + 1
        while not Text.objects.filter(pk=next_pk).exists() and next_pk < max_pk:
            next_pk += 1
        if not Text.objects.filter(pk=next_pk).exists():
            next_pk = None

        prev_pk = poem.pk - 1
        while not Text.objects.filter(pk=prev_pk).exists() and prev_pk > 0:
            prev_pk -= 1
        if not Text.objects.filter(pk=prev_pk).exists():
            prev_pk = None
        context['next_pk'] = next_pk if next_pk is not None else poem.pk

        context['prev_pk'] = prev_pk if prev_pk is not None else poem.pk
        context['can_edit'] = self.request.user.is_superuser
        return context

def search(request):
    return render(request, 'search.html')

def home(request):
    return render(request, 'base.html')


def index2(request, text):

    obj = Text.objects.all()
    txts = []
    for i in obj:
        txts.append(i.text)
    return render(request, 'words/index.html', {'txts': txts})


def index(request):

    analyzer = AnalyzerDD()
    analyzer.load_model(os.path.join('kaznlp', 'morphology', 'mdl'))
    mdl = os.path.join('kaznlp', 'tokenization', 'tokhmm.mdl')

    tagger = TaggerHMM(lyzer=analyzer)
    tagger.load_model(os.path.join('kaznlp', 'morphology', 'mdl'))
    text = request.GET.get('q', '')
    # text = u'Еңбек етсең ерінбей, тояды қарның тіленбей.'
    
    tokenizer = TokenizerHMM(model=mdl)
    result = dict()
    for sentence in tokenizer.tokenize(text):
        lower_sentence = map(lambda x: x.lower(), sentence)
        for i, a in enumerate(tagger.tag_sentence(lower_sentence)):
            result[sentence[i]]= {i.split('_')[0]: '_'.join(i.split('_')[1:]) for i in a.split()}

    return render(request, 'search.html', {'text': text, 'result': result, 'query': True})