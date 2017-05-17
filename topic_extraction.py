#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId
from utils import strip_html_tags, extract_urls
from watson import analyze_text, analyze_url

client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def parse_post(post):
    # Extract text and strip html tags and links
    content = strip_html_tags(post['rawContent'])
    links = extract_urls(content)

    for url in links:
        content = content.replace(url, '')

    try:
        images = list(map(lambda x: x['cdnUrl'], post['entities']['images']))
    except KeyError:
        images = []

    return content, links, images


query = {'type': 'POST'}
projection = {'rawContent': 1, 'entities': 1}

for post in hubchat.comments.find(query, projection, limit=10):
    text, urls, images_urls = parse_post(post)

    keywords = []
    categories = []

    # Analise text
    result = analyze_text(text)

    if result:
        keywords += result["keywords"]
        categories += result["categories"]

    # Analise links
    for url in urls:
        result = analyze_url(url)
        if result:
            keywords += result["keywords"]
            categories += result["categories"]

    # Write to mongo
    for keyword in keywords:
        hubchat.postkeywords.insert_one({
            'post': ObjectId(post['_id']),
            'text': keyword['text'],
            'relevance': keyword['relevance']
        })

    for category in categories:
        hubchat.postcategories.insert_one({
            'post': ObjectId(post['_id']),
            'text': category['label'],
            'relevance': category['score']
        })
