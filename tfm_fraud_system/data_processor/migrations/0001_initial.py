# Generated by Django 3.2.13 on 2022-06-18 23:38

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('cc_number', models.CharField(max_length=20)),
                ('first_name', models.CharField(max_length=255)),
                ('last_name', models.CharField(max_length=255)),
                ('gender', models.CharField(max_length=1)),
                ('row_number', models.IntegerField()),
                ('street', models.CharField(max_length=255)),
                ('city', models.CharField(max_length=255)),
                ('job', models.CharField(max_length=255)),
                ('profile', models.CharField(max_length=255)),
            ],
        ),
    ]
