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
        return 5
    elif LOW_MID <= user_like_rate < MID_HIGH:
        return 4
    elif MID_HIGH <= user_like_rate <= 1.0:
        return 3


def comments_base(comments, comments_avg):
    if comments == 0:
        return 1

    # above or below avg
    if comments <= comments_avg:
        importance = 0
    else:
        importance = 1

    if comments == 1:
        return 2 + importance
    elif 2 <= comments <= 3:
        return 3 + importance
    elif 4 <= comments <= 5:
        return 4 + importance
    elif comments >= 5:
        return 5


def comments_sum(comments, comments_avg):
    if comments == 0:
        return 0

    # above or below avg
    if comments <= comments_avg:
        return 1
    else:
        return 2


def calculate_rating(postID, postUUID, userID, user_like_rate, user_comments_avg):

    liked = did_like(postID, userID)
    comments = get_comments(postUUID, userID)
    avg = int(round(user_comments_avg))

    if liked:
        score = likes_base(user_like_rate) + comments_sum(comments, avg)
        return score if score <= 5 else 5
    else:
        return comments_base(comments, avg)


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

        score = calculate_rating(postID, postUUID, userID, user_like_rate, user_comments_avg)

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


def update_comments_avg(userID, avg):
    hubchat.users.update(
        {
            "_id": ObjectId(userID)
        },
        {
            "$set": {
                "commentsAvg": avg
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


def get_comment_avg(userID, postUUIDs):

    sum = 0.0
    for uuid in postUUIDs:
        sum += get_comments(uuid, userID)

    if len(postUUIDs) == 0.0:
        return 0.0

    return sum / len(postUUIDs)


def calculate_comments_average():

    comments_avg = []

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

        avg = get_comment_avg(userID, postUUIDs)

        update_comments_avg(userID, avg)

        if avg != 0.0:
            comments_avg.append(avg)

    comments_avg = sorted(comments_avg)

    fig, ax = plt.subplots()
    ax.hist(comments_avg, bins=50)

    ax.set_xlabel('comments avg')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("./comments_avg.pdf", format='pdf')

    return comments_avg

