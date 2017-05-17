import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1, WatsonException
import watson_developer_cloud.natural_language_understanding.features.v1 as features


natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='bafeb5b9-cf1d-49c3-82df-a93551260b16',
    password='mpkiBv0WJTRt')


def analyze_text(text):
    try:
        response = natural_language_understanding.analyze(
            text=text,
            features=[features.Keywords(), features.Categories()])
    except WatsonException:
        return None

    return response


def analyze_url(url):
    try:
        response = natural_language_understanding.analyze(
            url=url,
            features=[features.Keywords(), features.Categories()])
    except WatsonException:
        return None

    return response
