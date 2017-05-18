import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1, VisualRecognitionV3, WatsonException
import watson_developer_cloud.natural_language_understanding.features.v1 as features


natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='bafeb5b9-cf1d-49c3-82df-a93551260b16',
    password='mpkiBv0WJTRt')

visual_recognition = VisualRecognitionV3('2016-05-20', api_key='88b51a04aa6e0c88a816d541fbdae548cfc73629')


def analyze_text(text):
    try:
        response = natural_language_understanding.analyze(
            text=text,
            features=[features.Keywords(), features.Categories()])
    except WatsonException as e:
        print("analyze_text: WatsonException")
        if 'Error: limit exceeded for free plan, Code: 403' in e.args:
            raise e
        return None

    return response


def analyze_url(url):
    try:
        response = natural_language_understanding.analyze(
            url=url,
            features=[features.Keywords(), features.Categories()])
    except WatsonException as e:
        print("analyze_url: WatsonException")
        if 'Error: limit exceeded for free plan, Code: 403' in e.args:
            raise e
        return None

    return response


def analyze_image(url):
    try:
        result = visual_recognition.classify(images_url=url)
        return result['images'][0]['classifiers'][0]['classes']
    except WatsonException as e:
        print("analyze_image: WatsonException")
        if 'Error: limit exceeded for free plan, Code: 403' in e.args:
            raise e
        return None
    except (KeyError, TypeError):
        print("analyze_image: Key error in parsing image results")
        return None
