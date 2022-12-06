from django.db import models
from django.urls import reverse
# Create your models here.

class Theme(models.Model):
    theme = models.CharField("Жанр", max_length=50, blank=False)

    class Meta:
        verbose_name = "Жанр"
        verbose_name_plural = "Жанрлар"

    def __str__(self):
        return 'Жанр: ' + str(self.theme)


class Text(models.Model):
    text = models.TextField("Мәтін", blank=False)
    name = models.CharField("Мәтін атауы", max_length=50, blank=True)
    author = models.CharField("Автор", max_length=50, blank=True)
    themes = models.ManyToManyField(Theme, verbose_name="Жанр", blank=True)

    # is_restricted = models.BooleanField("Стихи с ограниченным доступом", default=False)
    # is_standard = models.BooleanField("Эталонные (проверенные) стихи", default=False)
    class Meta:
        verbose_name = "Мәтін"
        verbose_name_plural = "Мәтін"
        # permissions = (
        #     ("can_view_restricted_poems", "Can view restricted poems"),
        # )
    def __str__(self):
        return 'Мәтін: ' + self.get_name() + ", " + str(self.author)

    def get_name(self):
        name = self.name
        if name == "":
            name = self.text.strip().split("\n")[0]
            i = len(name) - 1
            while i > 0 and not name[i].isalpha():
                i -= 1
            name = name[:i+1]
        return name

    def get_name_short(self):
        name = self.__clean_name(self.name)
        if name == "":
            name = self.__get_name_form_text()
        return name[:64].replace(" ", "")

    def __get_name_form_text(self):
        name = ""
        line_number = 0
        while name == "":
            name = self.text.strip().split("\n")[line_number]
            i = len(name) - 1
            while i > 0 and not name[i].isalpha():
                i -= 1
            name = name[:i+1]
            name = self.__clean_name(name)
            line_number += 1
        return name

    def __clean_name(self, name):
        new_name = ""
        for ch in name:
            if ch.isalpha() or ch.isalnum() or ch == " ":
                new_name += ch
        return new_name.strip()

    def count_lines(self):
        return len(self.text.rstrip().split("\n"))

    def count_automatic_errors(self):
        for markup in self.markups.all():
            if "Automatic" in markup.author:
                return markup.get_automatic_additional().get_metre_errors_count()

    def get_absolute_url(self):
        if len(self.markups.all()) != 0:
            return self.markups.all()[0].get_absolute_url()
        return reverse("corpus:poems")

    def get_automatic_markup(self):
        for markup in self.markups.all():
            if "Automatic" in markup.author:
                return markup
        return None

    def count_manual_markups(self):
        return sum([int(markup.markup_version.name == "Manual") for markup in self.markups.all()])


class Words(models.Model):

    word = models.CharField(max_length=200)
    describtion = models.CharField(max_length=600, null=True)
    
    def __str__(self):
        return self.word

class Morthps(models.Model):

    name = models.CharField(max_length=200)
    describtion = models.CharField(max_length=600, null=True)

    def __str__(self):

        return self.name

class Annotate(models.Model):

    root_word = models.CharField(max_length=200)
    morphs = models.CharField(max_length=200, null=True)

    def __str__(self):
        return self.root_word
    