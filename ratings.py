#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
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


def get_n_comments(postUUID, userID):
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


def likes_modifier(like_rate):
    """
    Modifier of the rate depending from like_rate
    :param like_rate:
    :return:
    """
    LOW_MID = 0.0243496214265
    MID_HIGH = 0.0930735930736

    if 0.0 <= like_rate < LOW_MID:
        return 3
    elif LOW_MID <= like_rate < MID_HIGH:
        return 2
    elif MID_HIGH <= like_rate <= 1.0:
        return 1


def comments_rate_base(comments_rate):
    """
    Returns the base rate of a comment according to comment_rate
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


def comments_avg_modifier(n_comments, comments_avg):
    """
    Normalizes the avg over scale -1 to 1.
    n_comments < to avg -> return -1
    n_comments == to avg -> return 0
    n_comments > to avg -> return +1
    :param n_comments:
    :param comments_avg:
    :return:
    """
    min = 0.0
    max = comments_avg * 2.0

    a = -1
    b = 1

    modifier = (((b - a) * (n_comments - min)) / (max - min)) + a
    modifier = int(round(modifier))

    return modifier if modifier <= 1 else 1


def comments_base(n_comments, comments_avg, comments_rate):
    """
    Returns the rating given for the number of comments.
    Base (depending from comment_rate) + modifier (depending from # of comments avg)
    :param n_comments:
    :param comments_avg:
    :param comments_rate:
    :return:
    """
    # No comments, base case
    if n_comments == 0:
        return 1

    # At least one comment
    rate = comments_rate_base(comments_rate) + comments_avg_modifier(n_comments, comments_avg)

    return rate if rate <= 4 else 4


def calculate_rating(postID, postUUID, userID, user_like_rate, user_comments_avg, user_comment_rate):
    """
    Calculate rating
    :param postID:
    :param postUUID:
    :param userID:
    :param user_like_rate:
    :param user_comments_avg:
    :param user_comment_rate:
    :return:
    """
    liked = did_like(postID, userID)
    n_comments = get_n_comments(postUUID, userID)

    score = comments_base(n_comments, user_comments_avg, user_comment_rate)

    if liked:
        score += likes_modifier(user_like_rate)

    return score if score <= 4 else 4


def update_ratings():
    """
    Calculate scores for each post seen, and update ratings collection
    :return:
    """

    hubchat.ratings.delete_many({})

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

        # Extract information
        postID = post['post']
        postUUID = post['postUUID']
        userID = post['user']
        timestamp = post['updatedAt']
        user_like_rate = post['the_user'][0]['likeRate'] if 'likeRate' in post['the_user'][0] else 0.0
        user_comments_avg = post['the_user'][0]['commentsAvg'] if 'commentsAvg' in post['the_user'][0] else 0.0
        user_comment_rate = post['the_user'][0]['commentsRate'] if 'commentsRate' in post['the_user'][0] else 0.0

        # Calculate score
        score = calculate_rating(postID, postUUID, userID, user_like_rate, user_comments_avg, user_comment_rate)

        # Update ratings collection
        hubchat.ratings.insert_one(
            {
                "user": ObjectId(userID),
                "post": ObjectId(postID),
                "rate": score,
                "createdAt": timestamp
            }
        )


def get_like_rate(userID, posts):
    """
    Calculate like rate as likes/posts_seen
    :param userID:
    :param posts:
    :return:
    """
    seen_count = len(posts)
    like_count = 0.0

    for post in posts:
        liked = did_like(post, userID)

        if liked:
            like_count += 1.0

    return 0.0 if seen_count == 0.0 else like_count / seen_count


def update_rate(userID, rate):
    """
    Update likeRate value for give user
    :param userID:
    :param rate:
    :return:
    """
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
    """
    Update comment rate and avg values for given user
    :param userID:
    :param rate:
    :param avg:
    :return:
    """
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
    """
    Helper function to calculate ranges by deviding list in three
    :param rates:
    :return:
    """
    split = np.array_split(np.array(rates), 3)
    low_mid = np.average([split[0][-1], split[1][0]])
    mid_high = np.average([split[1][-1], split[2][0]])

    return low_mid, mid_high


def calculate_like_rates():
    """
    Calculate like rates of the users who have seen posts.
    Save histogram
    :return:
    """
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
    ax.hist(like_rates, bins=50)

    ax.set_xlabel('like rate')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("figures/like_rate.pdf", format='pdf')

    return like_rates


def get_comment_rate_and_avg(userID, postUUIDs):
    """
    Calculate comment rate and avg given user and seen posts
    :param userID:
    :param postUUIDs:
    :return:
    """
    if len(postUUIDs) == 0.0:
        return 0.0, 0.0

    sum = 0.0
    count = 0.0

    for uuid in postUUIDs:
        n_comments = get_n_comments(uuid, userID)
        if n_comments > 0:
            count += 1
            sum += n_comments

    if count == 0.0:
        return 0.0, 0.0

    comment_rate = count / len(postUUIDs)  # How many posts with 1+ comments / seen posts
    comment_avg = sum / count  # How many comments left in avg (in the commented posts)

    return comment_rate, comment_avg


def calculate_comment_rate():
    """
    Calculate comment rates and avg of every user who has seen posts.
    Saves histogram
    :return:
    """
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

    fig, ax = plt.subplots()
    ax.hist(comments_avg, bins=50)

    ax.set_xlabel('comments avg')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("figures/comments_avg.pdf", format='pdf')

    fig, ax = plt.subplots()
    ax.hist(comment_rate, bins=50)

    ax.set_xlabel('comment rate')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("figures/comment_rate.pdf", format='pdf')

    return comment_rate, comments_avg


def create_ratings_histogram():
    """
    Save ratings histogram
    :return:
    """
    ratings = []

    for rate in hubchat.ratings.find():
        ratings.append(rate['rate'])

    ratings = sorted(ratings)

    fig, ax = plt.subplots()
    ax.hist(ratings, bins=50)

    ax.set_xlabel('Ratings')
    ax.set_ylabel('# users per bin')
    fig.tight_layout()
    plt.savefig("figures/ratings.pdf", format='pdf')

    return ratings


# calculate_like_rates()
# calculate_comment_rate()
update_ratings()
#create_ratings_histogram()