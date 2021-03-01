FlowerDetector
==============

## Download images from Flickr
- Create API of Flickr, and then prepare keys as follows
```
cd config
mv flickr_sample.json flickr.json
```
Then, fill in the blanks of `api_key` and `secret_key`

- Execute the donwloading script
```
cd ../scripts
python3.6 download_pictures_from_flickr.py
```