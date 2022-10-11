# Generated by Django 4.1.1 on 2022-10-11 10:55

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Annotate",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("root_word", models.CharField(max_length=200)),
                ("morphs", models.CharField(max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Morthps",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=200)),
                ("describtion", models.CharField(max_length=600, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Text",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name="Words",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("word", models.CharField(max_length=200)),
                ("describtion", models.CharField(max_length=600, null=True)),
            ],
        ),
    ]
