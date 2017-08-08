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



# PHASE 1
# print("Start phase 1")
# split_ratings_training_testing()
# update_similarities(phase=1)
#
# split_ratings_training_validate()
#
# # Bin users
# binned_users = BinnedUsers()
# for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
#     positive_ratings = user['count']
#     binned_users.add(positive_ratings, user["_id"])
#
# alphas = [0, 0.25, 0.5, 0.75, 1.0]
# user_profile_versions = [1, 2]
# category_score_methods = [1, 2]
# categories_keywords_weights = [(60, 40), (70, 30), (80, 20)]
#
# model = {
#     '[1-5]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[5-10]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[10-20]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[20-30]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[30-50]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[50-70]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[70-100]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[100-150]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     },
#     '[150+]': {
#         'f_score': -1,
#         'parameters': {
#             'user_profile_v': None,
#             'alpha': None,
#             'cat_score_method': None,
#             'cat_key_weight': None,
#             'merge_method': None
#         },
#         'alternative_parameters': []
#     }
# }

# for alpha in alphas:
#     for user_profile_version in user_profile_versions:
#         for category_score_method in category_score_methods:
#             for categories_keywords_weight in categories_keywords_weights:
#
#                 for bin, users in binned_users.user_bins.items():
#
#                     f_scores_int = []
#                     f_scores_un = []
#
#                     for user in users:
#                         user_id = str(user)
#
#                         cf_recommendation_list = get_recommendations(user_id, alpha)
#
#                         rated_posts_training = get_rated_posts_sorted_by_date_training(user_id)
#
#                         userprofile = build_user_profile(rated_posts_training, version=user_profile_version)
#                         cb_recommendation_list = get_cb_recommendations(user_id, userprofile, category_score_method, categories_keywords_weight, phase=1)
#
#                         final_recommendations_intersection = set(cf_recommendation_list) & set(cb_recommendation_list)
#                         tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_intersection, phase=1)
#                         f_score_int = get_f_score(tp, fp, fn, tn)
#                         f_scores_int.append(f_score_int)
#
#                         final_recommendations_union = set(cf_recommendation_list) | set(cb_recommendation_list)
#                         tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_union, phase=1)
#                         f_score_un = get_f_score(tp, fp, fn, tn)
#                         f_scores_un.append(f_score_un)
#
#                     f_score_int = np.average(f_scores_int) if f_scores_int else 0
#                     f_score_un = np.average(f_scores_un) if f_scores_un else 0
#
#                     if f_score_int > f_score_un:
#                         merge_method = "intersection"
#                     elif f_score_int < f_score_un:
#                         merge_method = "union"
#                     else:
#                         merge_method = "indifferent"
#
#                     f_score_local_max = max(f_score_int, f_score_un)
#
#                     if f_score_local_max == model[bin]['f_score']:
#                         model[bin]['alternative_parameters'].append({
#                             'alpha': alpha,
#                             'user_profile_v': user_profile_version,
#                             'cat_score_method': category_score_method,
#                             'cat_key_weight': categories_keywords_weight,
#                             'merge_method': merge_method
#                         })
#
#                     if f_score_local_max > model[bin]['f_score']:
#                         model[bin]['f_score'] = f_score_local_max
#                         model[bin]['parameters']['alpha'] = alpha
#                         model[bin]['parameters']['user_profile_v'] = user_profile_version
#                         model[bin]['parameters']['cat_score_method'] = category_score_method
#                         model[bin]['parameters']['cat_key_weight'] = categories_keywords_weight
#                         model[bin]['parameters']['merge_method'] = merge_method
#                         model[bin]['alternative_parameters'] = []
#
#
#                     print("%s[alpha=%s][user_pr_v=%s][cat_score_v=%s][cat_key=%s][merge_method=%s] -> f_score: %s" % (bin, alpha, user_profile_version, category_score_method, str(categories_keywords_weight), merge_method, f_score_local_max))
#
#
# print("Final model")
# print(model)
#
# exit()

print("Start phase 2")


update_similarities(phase=2)

merge_ratings_training_validate()

binned_users = BinnedUsers()
for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
    positive_ratings = user['count']
    binned_users.add(positive_ratings, user["_id"])


model = {'[30-50]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.6829640947288006, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}}, '[20-30]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.51587301587301593, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[5-10]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.66666666666666663, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[10-20]': {'alternative_parameters': [{'merge_method': 'intersection', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.43548085901027078, 'parameters': {'merge_method': 'intersection', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}}, '[100-150]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.53713527851458887, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[70-100]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.87121212121212122, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[1-5]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}], 'f_score': 0.044901065449010652, 'parameters': {'merge_method': 'union', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[50-70]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}], 'f_score': 0.63586956521739135, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[150+]': {'alternative_parameters': [{'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0, 'parameters': {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}}

results = {}

all_f_scores = []
for bin, users in binned_users.user_bins_with_count.items():

    results[bin] = {
        'n_users': len(users),
        'avg_recommendations_per_user': None,
        'avg_f_score': None
    }

    f_scores = []
    avg_recommendations_n = []

    for user in users:
        user_id = str(user['id'])
        positive_ratings = user['ratings_count']

        alpha = model[bin]['parameters']['alpha'] if model[bin]['parameters']['alpha'] != "indifferent" else 0
        user_profile_v = model[bin]['parameters']['user_profile_v'] if model[bin]['parameters']['user_profile_v'] != "indifferent" else 1
        category_score_method = model[bin]['parameters']['cat_score_method'] if model[bin]['parameters']['cat_score_method'] != "indifferent" else 1
        categories_keywords_weight = model[bin]['parameters']['cat_key_weight'] if model[bin]['parameters']['cat_key_weight'] != "indifferent" else (60, 40)
        merge_method = model[bin]['parameters']['merge_method'] if model[bin]['parameters']['merge_method'] != "indifferent" else "union"

        # Calculate recommendations
        cf_recommendation_list = get_recommendations(user_id, alpha)

        rated_posts_training = get_rated_posts_sorted_by_date_training(user_id)

        userprofile = build_user_profile(rated_posts_training, version=user_profile_v)
        cb_recommendation_list = get_cb_recommendations(user_id, userprofile, category_score_method,
                                                        categories_keywords_weight, phase=2)

        final_recommendations_list = set(cf_recommendation_list) & set(
            cb_recommendation_list) if merge_method == 'intersection' else set(cf_recommendation_list) | set(
            cb_recommendation_list)

        tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_list, phase=2)
        f_score_partial = get_f_score(tp, fp, fn, tn)
        f_scores.append(f_score_partial)

        all_f_scores.append({
            'positive_ratings': positive_ratings,
            'f_score': f_score_partial
        })

        l = len(final_recommendations_list) if len(final_recommendations_list) else 0
        avg_recommendations_n.append(l)

    f_score_final = np.average(f_scores) if f_scores else 0

    results[bin]['avg_f_score'] = f_score_final
    results[bin]['avg_recommendations_per_user'] = np.average(avg_recommendations_n)

print('final results')
print(results)
print("all f-scores")
print(all_f_scores)