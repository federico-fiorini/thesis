import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as features


natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='bafeb5b9-cf1d-49c3-82df-a93551260b16',
    password='mpkiBv0WJTRt')

response = natural_language_understanding.analyze(
    url='https://www.graph.cool',
    features=[features.Entities(), features.Keywords(), features.Categories(), features.Concepts()])

print(json.dumps(response, indent=2))