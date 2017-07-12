#!.env/bin/python3

from user_profile import *
import numpy as np

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


def split_training_validate_testing(ratings):

    testing_len = round(len(ratings) * 10 / 100) if len(ratings) >= 10 else 1
    testing_set = ratings[-testing_len:]
    remaining_set = ratings[:-testing_len]

    validate_len = round(len(remaining_set) * 10 / 100) if len(remaining_set) >= 10 else 1
    validate_set = remaining_set[-validate_len:]
    training_set = remaining_set[:-validate_len]

    return training_set, validate_set, testing_set


def get_confusion_matrix(userprofile, testing_set, category_score_method, categories_keywords_weight):

    tp = 0.0  # True positive
    fp = 0.0  # False positive
    fn = 0.0  # False negative
    tn = 0.0  # True negative

    for rate in testing_set:
        postprofile = rate['postprofile']
        score = predict_score(userprofile, postprofile, category_method=category_score_method,
                              cat_key_weight=categories_keywords_weight)

        to_recommend = True if 3 <= score <= 4 else False
        if to_recommend:
            if 3 <= int(rate['rate']) <= 4:
                tp += 1
            else:
                fp += 1
        else:
            if 3 <= int(rate['rate']) <= 4:
                fn += 1
            else:
                tn += 1

    return tp, fp, fn, tn

# PHASE 1

user_profile_versions = [1, 2]
category_score_methods = [1, 2]
categories_keywords_weights = [(50, 50), (60, 40), (70, 30), (80, 20)]


for user_profile_version in user_profile_versions:
    for category_score_method in category_score_methods:
        for categories_keywords_weight in categories_keywords_weights:

            true_positive = 0.0
            false_positive = 0.0
            false_negative = 0.0
            true_negative = 0.0

            recommendation_rate = {}  # How many recommendations over how many positive rates

            for user in hubchat.ratings.aggregate(users_with_positive_ratings_order_by_count):

                rated_posts = get_rated_posts_sorted_by_date(user["_id"])

                training_set, validate_set, _ = split_training_validate_testing(rated_posts)

                # Build user profile with training set
                userprofile = build_user_profile(training_set, version=user_profile_version)

                tp, fp, fn, tn = get_confusion_matrix(userprofile, validate_set, category_score_method, categories_keywords_weight)

                true_positive += tp
                false_positive += fp
                false_negative += fn
                true_negative += tn

                positive_ratings = user['count']
                if positive_ratings <= 1000:
                    try:
                        recommendation_rate[positive_ratings].append(tp)
                    except KeyError:
                        recommendation_rate[positive_ratings] = [tp]

            # Calculate total performances metrics
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)

            # Average recommendations per positive ratings
            recommendation_rate = {k: np.average(v) for k, v in recommendation_rate.items()}
            print(sorted(recommendation_rate.items(), key=lambda t: t[0]))

            print("[Validation][User profile v=%s][Category score v=%s][Cat/Key weight=%s] Precision: %s , Recall: %s , Accuracy: %s"
                  % (user_profile_version, category_score_method, str(categories_keywords_weight), precision, recall, accuracy))

            print("Correct recommendations: %s" % true_positive)