#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient('localhost', 28017)
hubchat = client.hubchat

for user in hubchat.users.find({"_id": ObjectId("56a92f8b6675144e00c8f0dc")}):

    categories_profile = {}
    keywords_profile = {}

    for rate in hubchat.ratings.aggregate([
        {
            "$match": {"user": user["_id"]}
        },
        {
            "$lookup": {
                "from": "postcategories",
                "localField": "post",
                "foreignField": "post",
                "as": "postcategories"
            }
        },
        {
            "$lookup": {
                "from": "postkeywords",
                "localField": "post",
                "foreignField": "post",
                "as": "postkeywords"
            }
        }
    ]):

        for category in rate['postcategories']:
            score = (float(rate['rate']) - 2) * float(category['relevance'])

            try:
                categories_profile[category['text']] += score
            except KeyError:
                categories_profile[category['text']] = score

        for keyword in rate['postkeywords']:
            score = (float(rate['rate']) - 2) * float(keyword['relevance'])

            try:
                keywords_profile[keyword['text']] += score
            except KeyError:
                keywords_profile[keyword['text']] = score

    # Keep only positive
    categories_profile = {k: v for k, v in categories_profile.items() if v > 0.0}
    keywords_profile = {k: v for k, v in keywords_profile.items() if v > 0.0}

    print(categories_profile)
    print(keywords_profile)
