#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId
from utils import strip_html_tags, extract_urls
from watson import analyze_text, analyze_url, analyze_image

client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def parse_post(post):
    # Extract text and strip html tags and links
    content = strip_html_tags(post['the_post']['rawContent'])
    links = extract_urls(content)

    for url in links:
        content = content.replace(url, '')

    try:
        images = list(map(lambda x: x['cdnUrl'], post['the_post']['entities']['images']))
    except KeyError:
        images = []

    return content, links, images

nlp_count = 0
image_count = 0

# query = {'type': 'POST', 'isAnalysed': "false"}
# projection = {'rawContent': 1, 'entities': 1}
#
# for post in hubchat.comments.find(query, projection):

for post in hubchat.ratings.aggregate([
    {
        "$sort": {
            "rate": -1
        }
    },
    {
        "$lookup": {
            "from": "comments",
            "localField": "post",
            "foreignField": "_id",
            "as": "the_post"
        }
    },
    {
        "$project": {
            "post": 1,
            "the_post": {
                "$arrayElemAt": ["$the_post", 0]
            }
        }
    },
    {
        "$match": {
            "the_post.isAnalysed": "false"
        }
    }
]):
    print("Post: " + str(post['post']))
    text, urls, images_urls = parse_post(post)
    print("Post parsed")

    if len(urls) + 1 + nlp_count > 1000:
        print("NLP API limit reached")
        break

    if len(images_urls) + image_count > 250:
        print("Visual recognition API limit reached")
        break

    post_id = ObjectId(post['post'])
    keywords = []
    image_keywords = []
    categories = []

    # Analise text
    result = analyze_text(text)
    nlp_count += 1
    print("Text analised")

    if result:
        if "keywords" in result:
            keywords += result["keywords"]
        if "categories" in result:
            categories += result["categories"]

    # Analise links
    for url in urls:
        result = analyze_url(url)
        print("Url analised")
        nlp_count += 1
        if result:
            if "keywords" in result:
                keywords += result["keywords"]
            if "categories" in result:
                categories += result["categories"]

    # Analise images
    for image in images_urls:
        result = analyze_image(image)
        print("Image analised")
        image_count += 1
        if result:
            image_keywords += result

    # Write to mongo
    for keyword in keywords:
        hubchat.postkeywords.insert_one({
            'post': post_id,
            'text': keyword['text'],
            'relevance': keyword['relevance']
        })

    for keyword in image_keywords:
        hubchat.postkeywords.insert_one({
            'post': post_id,
            'text': keyword['class'],
            'relevance': keyword['score']
        })

    for category in categories:
        hubchat.postcategories.insert_one({
            'post': post_id,
            'text': category['label'],
            'relevance': category['score']
        })

    print("Updating post")
    hubchat.comments.update(
        {
            "_id": post_id
        },
        {
            "$set": {
                "isAnalysed": "true"
            }
        }
    )