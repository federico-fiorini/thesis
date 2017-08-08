#!.env/bin/python3

from collaborative_filtering import *
from user_profile import get_rated_posts_sorted_by_date, get_not_rated_posts_sorted_by_date, get_rated_posts_sorted_by_date_testing, build_user_profile, predict_score, get_rated_posts_sorted_by_date_validate
from utils import f_score, RecommendationsBinAvg, BinnedUsers
import numpy as np


def get_confusion_matrix(user_id, recommendation_list, phase):

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    if recommendation_list is None:
        recommendation_list = set()

    for post_id in recommendation_list:

        query = {
            "user": ObjectId(user_id),
            "post": ObjectId(post_id)
        }

        rate = hubchat.ratings_validate.find_one(query) if phase == 1 else hubchat.ratings_testing.find_one(query)

        if rate is None:
            continue  # Ignore

        if 3 <= rate['rate'] <= 4:
            tp += 1  # In recommendation list and in testing set with rate >= 3
        elif 1 <= rate['rate'] <= 2:
            fp += 1  # In recommendation list and in testing set with rate < 3

    testing_rates = hubchat.ratings_validate.find({"user": ObjectId(user_id)}) if phase == 1 else hubchat.ratings_testing.find({"user": ObjectId(user_id)})

    for rate in testing_rates:

        if str(rate['post']) in recommendation_list:
            continue  # Already analysed

        if 3 <= rate['rate'] <= 4:
            fn += 1  # In testing set with rate >= 3 but not in recommendation list
        elif 1 <= rate['rate'] <= 2:
            tn += 1  # In testing set with rate < 3 and not in recommendation list

    return tp, fp, fn, tn


def get_cb_recommendations(user_id, userprofile, category_score_method, categories_keywords_weight, phase):
    cb_recommendation_list = []
    rated_posts = get_rated_posts_sorted_by_date_validate(user_id) if phase == 1 else get_rated_posts_sorted_by_date_testing(user_id)
    for rate in rated_posts:
        postprofile = rate['postprofile']
        score = predict_score(userprofile, postprofile, category_method=category_score_method,
                              cat_key_weight=categories_keywords_weight)

        if 3 <= score <= 4:
            cb_recommendation_list.append(str(rate['post']))

    return cb_recommendation_list


def get_cb_recommendations_all(user_id, userprofile, category_score_method, categories_keywords_weight):
    cb_recommendation_list = []
    unrated_posts = get_not_rated_posts_sorted_by_date(user_id)
    for post in unrated_posts:
        postprofile = post['postprofile']
        score = predict_score(userprofile, postprofile, category_method=category_score_method,
                              cat_key_weight=categories_keywords_weight)

        if 3 <= score <= 4:
            cb_recommendation_list.append(str(post['post']))

    return cb_recommendation_list


def get_f_score(tp, fp, fn, tn):
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        precision = 0.0
        recall = 0.0

    return f_score(precision, recall)


users_with_positive_ratings_order_by_count = [
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
        "$sort": {
            "count": -1
        }
    }
]


def produce_recommendations():
    analyzed_users = set([str(x['user']) for x in hubchat.recommendations.find({}, {"user": 1, "_id": 0})])

    binned_users = BinnedUsers()
    for user in hubchat.ratings.aggregate(users_with_positive_ratings_order_by_count):
        positive_ratings = int(user['count'])
        if positive_ratings > 5 and str(user["_id"]) not in analyzed_users:
            binned_users.add(positive_ratings, user["_id"])

    model = {'[30-50]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.6829640947288006, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}}, '[20-30]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.51587301587301593, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[5-10]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.66666666666666663, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[10-20]': {'alternative_parameters': [{'merge_method': 'intersection', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.43548085901027078, 'parameters': {'merge_method': 'intersection', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}}, '[100-150]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.53713527851458887, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[70-100]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.87121212121212122, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[1-5]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}], 'f_score': 0.044901065449010652, 'parameters': {'merge_method': 'union', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[50-70]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}], 'f_score': 0.63586956521739135, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[150+]': {'alternative_parameters': [{'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0, 'parameters': {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}}

    for bin, users in binned_users.user_bins_with_count.items():

        for user in users:
            user_id = str(user['id'])
            positive_ratings = user['ratings_count']

            alpha = model[bin]['parameters']['alpha'] if model[bin]['parameters']['alpha'] != "indifferent" else 0
            user_profile_v = 1
            category_score_method = 1
            categories_keywords_weight = (60, 40)
            merge_method = "union"

            # Calculate recommendations
            cf_recommendation_list = get_recommendations_all(user_id, alpha)

            rated_posts = get_rated_posts_sorted_by_date(user_id)

            userprofile = build_user_profile(rated_posts, version=user_profile_v)
            cb_recommendation_list = get_cb_recommendations_all(user_id, userprofile, category_score_method,
                                                            categories_keywords_weight)

            final_recommendations_list = set(cf_recommendation_list) | set(cb_recommendation_list)
            hubchat.recommendations.insert_one(
                {
                    "user": ObjectId(user_id),
                    "positive_ratings": positive_ratings,
                    "recommendation_list": list(final_recommendations_list)
                }
            )

for user in hubchat.recommendations.find({}):
    user_id = user['user']
    recommendation_list = user['recommendation_list']

    if len(recommendation_list) == 0:
        continue

    positive_ratings = user['positive_ratings']

    user_communities = set([x['forum'] for x in hubchat.forummembers.find({"user": user_id}, {"forum": 1})])

    existing = 0.0
    new = 0.0

    for post in recommendation_list:
        post_ = hubchat.comments.find_one({"_id": ObjectId(post)}, {"forum": 1})
        if post_['forum'] in user_communities:
            existing += 1
        else:
            new += 1

    # print("User %s with %s positive ratings has %s recommendations: %s to existing comm and %s to new"
    #       % (user_id, positive_ratings, len(recommendation_list), existing, new))

    existing_per = existing / len(recommendation_list) * 100.0
    new_per = new / len(recommendation_list) * 100.0

    print("Pos ratings: %s , Existing: %s , New: %s" % (positive_ratings, existing_per, new_per))