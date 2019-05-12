import cv2
import os
import sys
import imutils
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import main

import book_functions
import spine_segmentation as seg
import text_detection as td
import scraper


def evaluate_performace(submission_id = "000000000"):
    
    fp = open(main.BASE_PATH + "/data/predicted_titles.txt", "w")

    dir_path = os.path.join(main.BASE_PATH, "data/training")
    spine_image_names = sorted(os.listdir(dir_path), key = lambda k : int(k.split('.')[0]))

    books = []

    for image_name in spine_image_names:
        print(image_name)
        fp.write(image_name + "\n")

        img_path = os.path.join(dir_path, image_name)

        img = cv2.imread(img_path)

        #lines, img = seg.get_book_lines(img_path, submission_id, debug = False)

        #spines = book_functions.get_spines_from_lines(img, lines, submission_id, debug = False)

        spines = [book_functions.Spine(img, [(0, 0), (0, img.shape[0]-1), (img.shape[1]-1, 0), (img.shape[1]-1, img.shape[0]-1)])]

        texts = td.text_detection_multi_image(spines, submission_id, debug = False)

        #print(texts)

        spine_words = []

        for bounding_box, text in texts[0]:
            #print(bounding_box, text)
            spine_words.append(book_functions.Word(text, bounding_box))

        spine = spines[0]
        spine.set_spine_words(spine_words)
        
        #print(spine.sentence)


        # Run the scraping pipeline for each spine
        books = []

        #print([word.string for word in spine.words])

        book_info = {}

        # Get query
        search_query = spine.sentence

        print(search_query)

        search_url = scraper.get_google_search_url_from_query(search_query)
        print(search_url)

        # Get first amazon link from google search url
        amazon_url = scraper.get_amazon_url_from_google_search(search_url)
        print(amazon_url)

        if amazon_url != None:
            isbn10 = scraper.get_isbn10_from_amazon_url(amazon_url, debug = True)

        else:
            # Couldn't get isbn10 from amazon link (or there was no amazon link)
            isbn10 = None

        if not scraper.is_isbn10(isbn10):
            print(isbn10, "is not a right ISBN.")
            #continue

        book_info, content = scraper.query_google_books_api(isbn10, type = "isbn", debug = False)
        book_info['api'] = 'Amazon and Google Books API'

        #print(book_info)
        if(content["totalItems"] == 0):
            book_info, content = scraper.query_google_books_api(search_query, type = "title", debug = False)
            book_info['api'] = 'Google Books API'

        print(book_info)

        if("title" in book_info):
            fp.write(book_info["title"] + "\n")
        else:
            fp.write("Recognized Text: " + search_query + "\n")

        if("authors" in book_info):
            fp.write(','.join(book_info["authors"]) + "\n")

        fp.write("\n")

        book = book_functions.Book(book_info, spine)

        books.append(book)

    fp.close()

    return books


evaluate_performace()