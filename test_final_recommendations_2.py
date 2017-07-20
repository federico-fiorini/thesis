#!.env/bin/python3

from collaborative_filtering import *
from user_profile import get_rated_posts_sorted_by_date_training, get_rated_posts_sorted_by_date_testing, build_user_profile, predict_score, get_rated_posts_sorted_by_date_validate
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


# split_ratings_training_testing()
#
# # PHASE 1
# print("Start phase 1")
# update_similarities(phase=1)
#
# split_ratings_training_validate()


alphas = [0, 0.25, 0.5, 0.75, 1.0]
user_profile_versions = [1, 2]
category_score_methods = [1, 2]
categories_keywords_weights = [(60, 40), (70, 30), (80, 20)]

model = {
    'parameters': {},
    'f_scores': None
}
f_score_max = 0.0

for alpha in alphas:
    for user_profile_version in user_profile_versions:
        for category_score_method in category_score_methods:
            for categories_keywords_weight in categories_keywords_weights:

                f_scores_int = []
                f_scores_un = []

                for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):

                    user_id = str(user['_id'])
                    positive_ratings = user['count']

                    cf_recommendation_list = get_recommendations(user_id, alpha)

                    rated_posts_training = get_rated_posts_sorted_by_date_training(user_id)

                    userprofile = build_user_profile(rated_posts_training, version=user_profile_version)
                    cb_recommendation_list = get_cb_recommendations(user_id, userprofile, category_score_method,
                                                                    categories_keywords_weight, phase=1)

                    final_recommendations_intersection = set(cf_recommendation_list) & set(cb_recommendation_list)
                    tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_intersection, phase=1)
                    f_score_int = get_f_score(tp, fp, fn, tn)

                    f_scores_int.append({
                        'positive_ratings': positive_ratings,
                        'f_score': f_score_int
                    })

                    final_recommendations_union = set(cf_recommendation_list) | set(cb_recommendation_list)
                    tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_union, phase=1)
                    f_score_un = get_f_score(tp, fp, fn, tn)

                    f_scores_un.append({
                        'positive_ratings': positive_ratings,
                        'f_score': f_score_un
                    })

                f_score_avg_int = np.average([x['f_score'] for x in f_scores_int])
                f_score_avg_un = np.average([x['f_score'] for x in f_scores_un])

                if f_score_avg_int > f_score_avg_un:
                    merge_method = "intersection"
                    f_scores = f_scores_int.copy()
                elif f_score_avg_int < f_score_avg_un:
                    merge_method = "union"
                    f_scores = f_scores_un.copy()
                else:
                    merge_method = "indifferent"
                    f_scores = f_scores_un.copy()

                f_score_avg_max = max(f_score_avg_int, f_score_avg_un)
                if f_score_avg_max > f_score_max:
                    f_score_max = f_score_avg_max
                    model['parameters']['alpha'] = alpha
                    model['parameters']['user_profile_version'] = user_profile_version
                    model['parameters']['category_score_method'] = category_score_method
                    model['parameters']['categories_keywords_weight'] = categories_keywords_weight
                    model['parameters']['merge_method'] = merge_method
                    model['f_scores'] = f_scores

                print("[a=%s][user_profile_v=%s][cat_score_m=%s][cat_key_weight=%s][merge=%s] F-score: %s" %
                      (alpha, user_profile_version, category_score_method, categories_keywords_weight,
                       merge_method, f_score_avg_max))

print(model)
