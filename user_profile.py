#!.env/bin/python3

import random
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient('localhost', 28017)
hubchat = client.hubchat

TRAINING_TESTING_RATIO = 100 / 100


def category_distance(cat1, cat2):
    """
    Calculate distance between categories and subcategories
    Return -1 if categories have different roots
    :param cat1:
    :param cat2:
    :return:
    """
    if cat1 == cat2:
        return 0

    cat1_arr = cat1.strip('/').split('/')
    cat2_arr = cat2.strip('/').split('/')

    min_len = min(len(cat1_arr), len(cat2_arr))
    matches = 0

    for i in range(min_len):
        if cat1_arr[i] == cat2_arr[i]:
            matches += 1

    if matches == 0:
        return -1

    cat1_arr = cat1_arr[matches:]
    cat2_arr = cat2_arr[matches:]

    return len(cat1_arr) + len(cat2_arr)


def get_root(category):
    return category.strip('/').split('/')[0]


def split_by_root_category(categories):

    root_categories = {}

    for category, score in categories.items():
        root = get_root(category)
        try:
            root_categories[root][category] = score
        except KeyError:
            root_categories[root] = {}
            root_categories[root][category] = score

    return root_categories


def user_item_distance(user_profile, item_profile):

    MISSING_KEYWORD_FACTOR = -0.0005  # or  -0.001
    MISSING_CATEGORY_FACTOR = -0.005

    # Split item profile
    post_categories = {}
    post_keywords = {}
    for profile in item_profile:
        if profile['type'] == 'keyword':
            post_keywords[profile['text']] = profile['relevance']
        elif profile['type'] == 'category':
            post_categories[profile['text']] = profile['relevance']

    # Get keywords score
    keywords_score = 0
    matching = 0
    for keyword, score in user_profile['keywords'].items():
        try:
            to_sum = post_keywords[keyword] * float(score)
            matching += 1
        except KeyError:
            to_sum = MISSING_KEYWORD_FACTOR * float(score)

        keywords_score += to_sum

    # Avg
    if matching > 0:
        keywords_score /= float(matching)

    # Get score from categories
    categories_score = 0
    root_categories = split_by_root_category(user_profile['categories'])

    # For each root group
    for root_category, categories in root_categories.items():

        category_sum = 0
        found_matching = False

        # For each item category
        for post_category, relevance in post_categories.items():

            # Skip if different root category
            if root_category != get_root(post_category):
                continue

            # For each category in root group, keep the one with min distance
            min_dist = None
            user_score = None
            for user_category, score in categories.items():
                distance = category_distance(user_category, post_category)
                if min_dist is None or distance < min_dist:
                    min_dist = distance
                    user_score = score

            category_sum += relevance * float(user_score) * (1 / (2.0 ** min_dist))
            found_matching = True

        if found_matching:
            categories_score += category_sum
        else:
            # Add missing score for each category in root group
            for user_category, score in categories.items():
                categories_score += MISSING_CATEGORY_FACTOR * float(score)

    return categories_score, keywords_score


for user in hubchat.users.find({"_id": ObjectId("56a92f8b6675144e00c8f0dc")}):

    categories_profile = {}
    keywords_profile = {}

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

    random.shuffle(ratings)

    training_set_len = round(len(ratings) * TRAINING_TESTING_RATIO)
    training_set = ratings[:training_set_len]
    testing_set = ratings[training_set_len:]

    for rate in training_set:
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

    # Sum   NO
    # categories_profile = {k: v['score'] for k, v in categories_profile.items()}
    # keywords_profile = {k: v['score'] for k, v in keywords_profile.items()}

    # Sum - only positive   Mmmmm ni
    # categories_profile = {k: v['score'] for k, v in categories_profile.items() if v['score'] > 0}
    # keywords_profile = {k: v['score'] for k, v in keywords_profile.items() if v['score'] > 0}

    # Average   OK
    # categories_profile = {k: v['score'] / v['count'] for k, v in categories_profile.items()}
    # keywords_profile = {k: v['score'] / v['count'] for k, v in keywords_profile.items()}

    # Average - only positive   OK
    # categories_profile = {k: v['score'] / v['count'] for k, v in categories_profile.items() if v['score'] > 0}
    # keywords_profile = {k: v['score'] / v['count'] for k, v in keywords_profile.items() if v['score'] > 0}

    # Weighted average     OKish
    # categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items()}
    # keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items()}

    # Weighted average - only positive  OK
    # categories_profile = {k: v['score'] / v['relevance_sum'] for k, v in categories_profile.items() if v['score'] > 0}
    # keywords_profile = {k: v['score'] / v['relevance_sum'] for k, v in keywords_profile.items() if v['score'] > 0}

    userprofile = {'categories': categories_profile, 'keywords': keywords_profile}

    result = []

    # Calculate distance
    for rate in training_set:
        postprofile = rate['postprofile']
        cat_distance, key_distance = user_item_distance(userprofile, postprofile)
        result.append((key_distance, rate['rate']))

    result = sorted(result, key=lambda x: x[1])

    for r in result:
        print(r)