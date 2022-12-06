from django.db import models

# Create your models here.
class Text(models.Model):

    text = models.TextField()

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
    