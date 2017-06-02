#!.env/bin/python3

import random
from pymongo import MongoClient
from bson.objectid import ObjectId
from content_based import user_item_distance

client = MongoClient('localhost', 28017)
hubchat = client.hubchat

TRAINING_TESTING_RATIO = 100 / 100


def build_user_profile(user_ratings):
    """
    Build user profile give user ratings
    :param user_ratings:
    :return:
    """
    categories_profile = {}
    keywords_profile = {}

    for rate in user_ratings:
        for profile in rate['postprofile']:
            score = (float(rate['rate']) - 2) * float(profile['relevance'])

            if profile['type'] == 'category':
                try:
                    categories_profile[profile['text']]['score'] += score
                    categories_profile[profile['text']]['count'] += 1
                    categories_profile[profile['text']]['relevance_sum'] += profile['relevance']
                except KeyError:
                    categories_profile[profile['text']] = {}
                    categories_profile[profile['text']]['score'] = score
                    categories_profile[profile['text']]['count'] = 1
                    categories_profile[profile['text']]['relevance_sum'] = profile['relevance']

            elif profile['type'] == 'keyword':
                try:
                    keywords_profile[profile['text']]['score'] += score
                    keywords_profile[profile['text']]['count'] += 1
                    keywords_profile[profile['text']]['relevance_sum'] += profile['relevance']
                except KeyError:
                    keywords_profile[profile['text']] = {}
                    keywords_profile[profile['text']]['score'] = score
                    keywords_profile[profile['text']]['count'] = 1
                    keywords_profile[profile['text']]['relevance_sum'] = profile['relevance']

    # Average   OK
    categories_profile = {k: v['score'] / v['count'] for k, v in categories_profile.items()}
    keywords_profile = {k: v['score'] / v['count'] for k, v in keywords_profile.items()}

    # Average - only positive   OK
    # categories_profile = {k: v['score'] / v['count'] for k, v in categories_profile.items() if v['score'] > 0}
    # keywords_profile = {k: v['score'] / v['count'] for k, v in keywords_profile.items() if v['score'] > 0}

    # Weighted average     OKish
    # categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items()}
    # keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items()}

    # Weighted average - only positive  OK
    # categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items() if v['score'] > 0}
    # keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items() if v['score'] > 0}

    return {'categories': categories_profile, 'keywords': keywords_profile}


# For each user build user:
# 1. Get ratings
# 2. Split into training and testing sets
# 3. Build user profile with training test
# 3. Test results with testing test
for user in hubchat.users.find({"_id": ObjectId("56a92f8b6675144e00c8f0dc")}):
    ratings = list(hubchat.ratings.aggregate([
        {
            "$match": {"user": user["_id"]}
        },
        {
            "$lookup": {
                "from": "postprofile",
                "localField": "post",
                "foreignField": "post",
                "as": "postprofile"
            }
        }
    ]))

    # Shuffle to get random sample
    random.shuffle(ratings)

    # Split into training and testing sets
    training_set_len = round(len(ratings) * TRAINING_TESTING_RATIO)
    training_set = ratings[:training_set_len]
    # testing_set = ratings[training_set_len:]  # TODO: change to proper testing set
    testing_set = training_set.copy()

    # Build user profile with training set
    userprofile = build_user_profile(training_set)

    # Calculate distance and check results
    result = []
    for rate in testing_set:
        postprofile = rate['postprofile']
        distance = user_item_distance(userprofile, postprofile)
        result.append((distance, rate['rate']))

    result = sorted(result, key=lambda x: x[0])

    for r in result:
        print(r)