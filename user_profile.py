#!.env/bin/python3

import random
from pymongo import MongoClient
from bson.objectid import ObjectId
from content_based import predict_score
from sklearn.model_selection import KFold
import numpy as np


client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def save_user_profile(user_id, user_profile):
    hubchat.userprofile.insert_one({
        'user': user_id,
        'keywords': user_profile['keywords'],
        'categories': user_profile['categories']
    })


def get_rated_posts_sorted_by_date(user_id):
    return list(hubchat.ratings.aggregate([
        {
            "$match": {"user": user_id}
        },
        {
            "$sort": {
                "createdAt": 1
            }
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


def build_user_profile(user_ratings, version=1):
    """
    Build user profile give user ratings
    :param user_ratings:
    :param version:
    :return:
    """
    if not 1 <= version <= 2:
        print("[Logic Error] Version %s is not supported" % version)

    categories_profile = {}
    keywords_profile = {}

    for rate in user_ratings:
        for profile in rate['postprofile']:

            # Pre-filtering: only positive ratings (2, 3 and 4)
            if version == 2 and int(rate['rate']) == 1:
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


    # Weighted average
    categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items()}
    keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items()}

    return {'categories': categories_profile, 'keywords': keywords_profile}

