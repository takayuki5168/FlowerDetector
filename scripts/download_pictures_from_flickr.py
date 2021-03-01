import os, sys, time

import flickrapi
import json

from urllib.request import urlretrieve
import ssl

current_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

json_load = json.load(open(current_dir + '/../config/flickr.json', 'r'))
flickr_api_key = json_load['api_key']
secret_key = json_load['secret_key']

ssl._create_default_https_context = ssl._create_unverified_context

def download_pictures(keyword='', per_page=10):
    data_dir = current_dir + '/../dataset/' + keyword

    flickr = flickrapi.FlickrAPI(flickr_api_key, secret_key, format='parsed-json')
    response = flickr.photos.search(
        text=keyword + ' flower',
        per_page=per_page,
        media='photos',
        sort='relevance',
        safe_search=1,
        extras='url_q, license'
    )
    photos = response['photos']

    try:
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        for photo in photos['photo']:
            url_q = photo['url_q']
            filepath = data_dir + '/' + photo['id'] + '.jpg'
            urlretrieve(url_q, filepath)
            time.sleep(0.1)

    except Exception as e:
        print(e)

if __name__ == '__main__':
    keywords = []
    with open(current_dir + '/../config/flower_species.txt') as f:
        for l in f.readlines():
            w = l.split(', ')
            keywords.append(w[0])

    for keyword in keywords:
        print("Download '{}'".format(keyword))
        download_pictures(keyword=keyword)
