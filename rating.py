#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def did_like(postID, userID):
    """
    Checks whether the user liked or not the post
    :param postID:
    :param userID:
    :return:
    """
    for _ in hubchat.commentactions.aggregate([
        {
            "$match": {
                "post": ObjectId(postID),
                "createdBy": ObjectId(userID),
                "actionType": "voteUp",
                "deletedAt": {
                    "$exists": "false"
                }
            }
        },
        {
            "$project": {
                "post": 1,
                "comment": 1,
                "cmp": {
                    "$cmp": ["$post", "$comment"]
                }
            }
        },
        {
            "$match": {"cmp": 0}
        }
    ]):
        # There is one "voteUp" action
        return True
    # There is no "voteUp" action
    return False


def get_comments(postUUID, userID):
    """
    Returns how many comments the user left on the post
    :param postID:
    :param userID:
    :return:
    """
    count = 0

    for _ in hubchat.comments.find(
        {
            "postUUID": postUUID,
            "createdBy": ObjectId(userID),
        }
    ):
        count += 1

    return count


def likes_base(user_like_rate):
    LOW_MID = 0.0243496214265
    MID_HIGH = 0.0930735930736

    if 0.0 <= user_like_rate < LOW_MID:
        return 4
    elif LOW_MID <= user_like_rate < MID_HIGH:
        return 3
    elif MID_HIGH <= user_like_rate <= 1.0:
        return 2


def comments_rate_base(comments_rate):
    """

    :param comments_rate:
    :return:
    """
    LOW_MID = 0.16666666666666666
    MID_HIGH = 0.5

    if 0.0 <= comments_rate < LOW_MID:
        return 4
    elif LOW_MID <= comments_rate < MID_HIGH:
        return 3
    elif MID_HIGH <= comments_rate <= 1.0:
        return 2


def comments_avg_modifier(comments, comments_avg):
    min = 0.0
    max = comments_avg * 2.0

    a = -1
    b = 1

    modifier = (((b - a) * (comments - min)) / (max - min)) + a
    modifier = int(round(modifier))

    return modifier if modifier <= 1 else 1


def comments_base(comments, comments_avg, comments_rate):
    # No comments
    if comments == 0:
        return 1

    # At least one comment
    rate = comments_rate_base(comments_rate) + comments_avg_modifier(comments, comments_avg)

    return rate if rate <= 4 else 4


def comments_modifier(comments, comments_rate):

    if comments == 0:
        return 0

    return comments_rate_base(comments_rate) - 2


def calculate_rating(postID, postUUID, userID, user_like_rate, user_comments_avg, user_comment_rate):

    liked = did_like(postID, userID)
    comments = get_comments(postUUID, userID)

    if liked:
        score = likes_base(user_like_rate) + comments_modifier(comments, user_comment_rate)
        return score if score <= 4 else 4
    else:
        return comments_base(comments, user_comments_avg, user_comment_rate)


def update_ratings():
    scores = {}
    for post in hubchat.commentseens.aggregate([
        {
            "$lookup": {
                "from": "comments",
                "localField": "post",
                "foreignField": "_id",
                "as": "the_post"
            }
        },
        {
            "$lookup": {
                "from": "users",
                "localField": "user",
                "foreignField": "_id",
                "as": "the_user"
            }
        }
    ]):
        # Filter out results without the post
        if not post['the_post']:
            continue

        # Filter out comments
        if post['the_post'][0]['type'] != "POST":
            continue

        postID = post['post']
        postUUID = post['postUUID']
        userID = post['user']
        user_like_rate = post['the_user'][0]['likeRate'] if 'likeRate' in post['the_user'][0] else 0.0
        user_comments_avg = post['the_user'][0]['commentsAvg'] if 'commentsAvg' in post['the_user'][0] else 0.0
        user_comment_rate = post['the_user'][0]['commentsRate'] if 'commentsRate' in post['the_user'][0] else 0.0

        score = calculate_rating(postID, postUUID, userID, user_like_rate, user_comments_avg, user_comment_rate)

        hubchat.ratings.find_one_and_replace(
            {
                "user": ObjectId(userID),
                "post": ObjectId(postID)
            },
            {
                "user": ObjectId(userID),
                "post": ObjectId(postID),
                "rate": score
            },
            upsert=True
        )

        try:
            scores[score] += 1
        except KeyError:
            scores[score] = 1


    print(scores)

update_ratings()

def get_like_rate(userID, posts):

    seen_count = len(posts)
    like_count = 0.0

    for post in posts:
        liked = did_like(post, userID)

        if liked:
            like_count += 1.0

    return 0.0 if seen_count == 0.0 else like_count / seen_count


def update_rate(userID, rate):
    hubchat.users.update(
        {
            "_id": ObjectId(userID)
        },
        {
            "$set": {
                "likeRate": rate
            }
        }
    )


def update_comment_rate_and_avg(userID, rate, avg):
    hubchat.users.update(
        {
            "_id": ObjectId(userID)
        },
        {
            "$set": {
                "commentsAvg": avg,
                "commentsRate": rate,
            }
        }
    )


def get_ranges(rates):
    split = np.array_split(np.array(rates), 3)
    low_mid = np.average([split[0][-1], split[1][0]])
    mid_high = np.average([split[1][-1], split[2][0]])

    return low_mid, mid_high


def calculate_like_rates():

    like_rates = []

    for user in hubchat.commentseens.aggregate([
        {
            "$group": {
                "_id": "$user",
                "posts": {"$push": "$post"}
            }
        }
    ]):
        userID = user['_id']
        posts = user['posts']

        rate = get_like_rate(userID, posts)

        update_rate(userID, rate)

        if rate != 0.0:
            like_rates.append(rate)

    like_rates = sorted(like_rates)

    fig, ax = plt.subplots()
    ax.hist(like_rates, bins=20)

    ax.set_xlabel('like rate')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("./like_rate.pdf", format='pdf')

    return like_rates


def get_comment_rate_and_avg(userID, postUUIDs):

    if len(postUUIDs) == 0.0:
        return 0.0, 0.0

    sum = 0.0
    count = 0.0

    for uuid in postUUIDs:
        n_comments = get_comments(uuid, userID)
        if n_comments > 0:
            count += 1
            sum += n_comments

    if count == 0.0:
        return 0.0, 0.0

    comment_rate = count / len(postUUIDs)  # How many posts with 1+ comments / seen posts
    comment_avg = sum / count  # How many comments left in avg (in the commented posts)

    return comment_rate, comment_avg


def calculate_comment_rate():

    comments_avg = []
    comment_rate = []

    for user in hubchat.commentseens.aggregate([
        {
            "$lookup": {
                "from": "comments",
                "localField": "post",
                "foreignField": "_id",
                "as": "the_post"
            }
        },
        {
            "$group": {
                "_id": "$user",
                "posts": {"$push": "$the_post"}
            }
        }
    ]):
        userID = user['_id']

        postUUIDs = []
        for x in user['posts']:
            for post in x:
                postUUIDs.append(post['uuid'])

        rate, avg = get_comment_rate_and_avg(userID, postUUIDs)

        update_comment_rate_and_avg(userID, rate, avg)

        if avg != 0.0:
            comments_avg.append(avg)
            comment_rate.append(rate)

    comments_avg = sorted(comments_avg)
    comment_rate = sorted(comment_rate)

    print(comment_rate)
    print(comments_avg)

    fig, ax = plt.subplots()
    ax.hist(comments_avg, bins=50)

    ax.set_xlabel('comments avg')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("./comments_avg.pdf", format='pdf')

    fig, ax = plt.subplots()
    ax.hist(comment_rate, bins=50)

    ax.set_xlabel('comment rate')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("./comment_rate.pdf", format='pdf')

    return comment_rate, comments_avg
