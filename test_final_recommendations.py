#!.env/bin/python3

from collaborative_filtering import *
from user_profile import get_rated_posts_sorted_by_date_training, get_rated_posts_sorted_by_date_testing, build_user_profile, predict_score
from utils import f_score, RecommendationsBinAvg


def get_confusion_matrix(user_id, recommendation_list):

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    if recommendation_list is None or len(recommendation_list) == 0:
        return tp, fp, fn, tn

    for post_id in recommendation_list:

        query = {
            "user": ObjectId(user_id),
            "post": ObjectId(post_id)
        }

        rate = hubchat.ratings_testing.find_one(query)

        if rate is None:
            continue  # Ignore

        if 3 <= rate['rate'] <= 4:
            tp += 1  # In recommendation list and in testing set with rate >= 3
        elif 1 <= rate['rate'] <= 2:
            fp += 1  # In recommendation list and in testing set with rate < 3

    testing_rates = hubchat.ratings_testing.find({"user": ObjectId(user_id)})

    for rate in testing_rates:

        if str(rate['post']) in recommendation_list:
            continue  # Already analysed

        if 3 <= rate['rate'] <= 4:
            fn += 1  # In testing set with rate >= 3 but not in recommendation list
        elif 1 <= rate['rate'] <= 2:
            tn += 1  # In testing set with rate < 3 and not in recommendation list

    return tp, fp, fn, tn


def get_cb_recommendations(user, userprofile, category_score_method, categories_keywords_weight):
    cb_recommendation_list = []
    for rate in get_rated_posts_sorted_by_date_testing(user["_id"]):
        postprofile = rate['postprofile']
        score = predict_score(userprofile, postprofile, category_method=category_score_method,
                              cat_key_weight=categories_keywords_weight)

        if 3 <= score <= 4:
            cb_recommendation_list.append(str(rate['post']))

    return cb_recommendation_list


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

split_ratings_training_testing()

update_similarities(phase=2)

USER_PROFILE_VERSION = 2

alphas = [0, 0.25, 0.5, 0.75, 1.0]
category_score_methods = [1, 2]
categories_keywords_weights = [(50, 50), (60, 40), (70, 30), (80, 20)]

for alpha in alphas:
    for category_score_method in category_score_methods:
        for categories_keywords_weight in categories_keywords_weights:
            true_positive_int = 0.0
            false_positive_int = 0.0
            false_negative_int = 0.0
            true_negative_int = 0.0

            recommendations_avg_int = RecommendationsBinAvg()

            true_positive_un = 0.0
            false_positive_un = 0.0
            false_negative_un = 0.0
            true_negative_un = 0.0

            recommendations_avg_un = RecommendationsBinAvg()

            for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
                user_id = str(user['_id'])
                positive_ratings = user['count']

                cf_recommendation_list = get_recommendations(user_id, alpha)

                rated_posts_training = get_rated_posts_sorted_by_date_training(user["_id"])

                userprofile = build_user_profile(rated_posts_training, version=USER_PROFILE_VERSION)
                cb_recommendation_list = get_cb_recommendations(user, userprofile, category_score_method, categories_keywords_weight)

                final_recommendations_intersection = set(cf_recommendation_list) & set(cb_recommendation_list)
                tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_intersection)

                true_positive_int += tp
                false_positive_int += fp
                false_negative_int += fn
                true_negative_int += tn

                recommendations_avg_int.add(positive_ratings, len(final_recommendations_intersection))

                final_recommendations_union = set(cf_recommendation_list) | set(cb_recommendation_list)
                tp, fp, fn, tn = get_confusion_matrix(user_id, final_recommendations_union)

                true_positive_un += tp
                false_positive_un += fp
                false_negative_un += fn
                true_negative_un += tn

                recommendations_avg_un.add(positive_ratings, len(final_recommendations_union))

            precision_int = true_positive_int / (true_positive_int + false_positive_int)
            recall_int = true_positive_int / (true_positive_int + false_negative_int)
            f_score_int = f_score(precision_int, recall_int)

            precision_un = true_positive_un / (true_positive_un + false_positive_un)
            recall_un = true_positive_un / (true_positive_un + false_negative_un)
            f_score_un = f_score(precision_un, recall_un)

            recommendations_avg_int.avg()
            recommendations_avg_un.avg()

            print("[alpha=%s][cat_score_v=%s][cat_key=%s] Intersection -> precision: %s , recall: %s , f_score: %s | Union -> precision: %s , recall: %s , f_score: %s" % (alpha, category_score_method, str(categories_keywords_weight), precision_int, recall_int, f_score_int, precision_un, recall_un, f_score_un))
            print("[Avg number of recommendations per user (binned for positive ratings)] Intersection: %s , Union: %s" % (recommendations_avg_int, recommendations_avg_un))
            print("========")
