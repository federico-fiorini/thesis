#!.env/bin/python3

import random
from pymongo import MongoClient
from bson.objectid import ObjectId
from content_based import predict_score
from sklearn.model_selection import KFold
import numpy as np

client = MongoClient('localhost', 28017)
hubchat = client.hubchat

K_FOLD = 10


def build_user_profile(user_ratings, version=1):
    """
    Build user profile give user ratings
    :param user_ratings:
    :param version:
    :return:
    """
    if not 1 <= version <= 4:
        print("[Logic Error] Version %s is not supported" % version)

    categories_profile = {}
    keywords_profile = {}

    for rate in user_ratings:
        for profile in rate['postprofile']:

            # Pre-filtering: only positive ratings (2, 3 and 4)
            if (version == 2 or version == 4) and int(rate['rate']) == 1:
                continue

            score = float(rate['rate']) * float(profile['relevance'])

            if profile['type'] == 'category':
                try:
                    categories_profile[profile['text']]['score'] += score
                    # categories_profile[profile['text']]['count'] += 1
                    categories_profile[profile['text']]['relevance_sum'] += profile['relevance']
                except KeyError:
                    categories_profile[profile['text']] = {}
                    categories_profile[profile['text']]['score'] = score
                    # categories_profile[profile['text']]['count'] = 1
                    categories_profile[profile['text']]['relevance_sum'] = profile['relevance']

            elif profile['type'] == 'keyword':
                try:
                    keywords_profile[profile['text']]['score'] += score
                    # keywords_profile[profile['text']]['count'] += 1
                    keywords_profile[profile['text']]['relevance_sum'] += profile['relevance']
                except KeyError:
                    keywords_profile[profile['text']] = {}
                    keywords_profile[profile['text']]['score'] = score
                    # keywords_profile[profile['text']]['count'] = 1
                    keywords_profile[profile['text']]['relevance_sum'] = profile['relevance']

    # Average   OK
    # categories_profile = {k: v['score'] / v['count'] for k, v in categories_profile.items()}
    # keywords_profile = {k: v['score'] / v['count'] for k, v in keywords_profile.items()}

    # Average - only positive   OK
    # categories_profile = {k: v['score'] / v['count'] for k, v in categories_profile.items() if v['score'] > 0}
    # keywords_profile = {k: v['score'] / v['count'] for k, v in keywords_profile.items() if v['score'] > 0}

    if version == 1 or version == 2:
        # Weighted average
        categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items()}
        keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items()}

    else:
        # Weighted average - only positive
        categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items() if v['score'] / v['relevance_sum'] > 1}
        keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items() if v['score'] / v['relevance_sum'] > 1}

    return {'categories': categories_profile, 'keywords': keywords_profile}


# For each user build user:
# 1. Get ratings
# 2. Split into training and testing sets
# 3. Build user profile with training test
# 3. Test results with testing test
# for user in hubchat.users.find({"_id": ObjectId("57cd17445ed49f1c00d4a80f")}):
for user in hubchat.ratings.aggregate([
    {
        "$match": {
            "rate": {"$gt": 1}
        }
    },
    {
        "$group": {
            "_id": "$user",
            "count": {"$sum": 1}
        }
    },
    {
        "$match": {
            "count": {"$gt": 50}
        }
    }
]):
    rated_posts = list(hubchat.ratings.aggregate([
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
    random.shuffle(rated_posts)
    np_rated_posts = np.array(rated_posts)

    results = []
    kf = KFold(n_splits=K_FOLD)
    for train, test in kf.split(rated_posts):

        # Split into training and testing sets
        training_set = np_rated_posts[train]
        testing_set = np_rated_posts[test]

        # Build user profile with training set
        userprofile = build_user_profile(training_set, version=4)

        # Calculate distance and check results
        correct = 0
        incorrect = 0
        rec_correct = 0
        rec_incorrect = 0

        for rate in testing_set:
            postprofile = rate['postprofile']
            score = round(predict_score(userprofile, postprofile))

            if score == int(rate['rate']):
                correct += 1
            else:
                incorrect += 1

            recommend = True if 3 <= score <= 4 else False
            if not recommend:
                continue

            if 3 <= int(rate['rate']) <= 4:
                rec_correct += 1
            else:
                rec_incorrect += 1

        results.append({"correct_rate": correct, "incorrect_rate": incorrect, "correct_rec": rec_correct, "incorrect_rec": rec_incorrect})

    sum_correct = 0
    sum_incorrect = 0
    sum_rec_correct = 0
    sum_rec_incorrect = 0
    for result in results:
        sum_correct += result["correct_rate"]
        sum_incorrect += result["incorrect_rate"]
        sum_rec_correct += result["correct_rec"]
        sum_rec_incorrect += result["incorrect_rec"]

    print({"correct_rate": sum_correct/float(K_FOLD), "incorrect_rate": sum_incorrect/float(K_FOLD),
           "correct_rec": sum_rec_correct/float(K_FOLD), "incorrect_rec": sum_rec_incorrect/float(K_FOLD)})