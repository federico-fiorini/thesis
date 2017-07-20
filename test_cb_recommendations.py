#!.env/bin/python3

from user_profile import *
from utils import f_score, BinnedUsers
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


def get_f_score(tp, fp, fn, tn):
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        precision = 0.0
        recall = 0.0

    return f_score(precision, recall)

# PHASE 1

user_profile_versions = [1, 2]
category_score_methods = [1, 2]
categories_keywords_weights = [(60, 40), (80, 20)]

model = {
    '[1-5]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[5-10]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[10-20]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[20-30]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[30-50]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[50-70]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[70-100]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[100-150]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    },
    '[150+]': {
        'f_score': -1,
        'parameters': {
            'user_profile_v': None,
            'cat_score_method': None,
            'cat_key_weight': None,
        },
        'alternative_parameters': []
    }
}

# Bin users
binned_users = BinnedUsers()
for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
    positive_ratings = user['count']
    binned_users.add(positive_ratings, user["_id"])


for user_profile_version in user_profile_versions:
    for category_score_method in category_score_methods:
        for categories_keywords_weight in categories_keywords_weights:

            for bin, users in binned_users.user_bins.items():

                f_scores = []

                for user in users:
                    user_id = str(user)

                    rated_posts = get_rated_posts_sorted_by_date(user_id)

                    training_set, validate_set, _ = split_training_validate_testing(rated_posts)

                    # Build user profile with training set
                    userprofile = build_user_profile(training_set, version=user_profile_version)

                    tp, fp, fn, tn = get_confusion_matrix(userprofile, validate_set, category_score_method, categories_keywords_weight)
                    f_score_value = get_f_score(tp, fp, fn, tn)

                    f_scores.append(f_score_value)

                f_score_avg = np.average(f_scores) if f_scores else 0

                if f_score_avg == model[bin]['f_score']:
                    model[bin]['alternative_parameters'].append({
                        'user_profile_v': user_profile_version,
                        'cat_score_method': category_score_method,
                        'cat_key_weight': categories_keywords_weight
                    })

                if f_score_avg > model[bin]['f_score']:
                    model[bin]['f_score'] = f_score_avg
                    model[bin]['parameters']['user_profile_v'] = user_profile_version
                    model[bin]['parameters']['cat_score_method'] = category_score_method
                    model[bin]['parameters']['cat_key_weight'] = categories_keywords_weight

                print("%s[user_profile_v=%s][cat_score_m=%s][cat_key_weight=%s] F-score: %s" %
                      (bin, user_profile_version, category_score_method, categories_keywords_weight, f_score_avg))

print(model)
