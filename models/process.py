import cv2
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


def full_pipeline(img_path, submission_id):
    
    img_path = main.BASE_PATH + "/data/05.jpg"
    print(img_path)

    img = cv2.imread(img_path)

    lines, img = seg.get_book_lines(img_path, submission_id, debug = False)

    spines = book_functions.get_spines_from_lines(img, lines, submission_id, debug = False)

    texts = td.text_detection_multi_image(spines, submission_id, debug = False)

    #print(texts)

    for i in range(len(texts)):
        spine_words = []

        for bounding_box, text in texts[i]:
            #print(bounding_box, text)
            spine_words.append(book_functions.Word(text, bounding_box))

        spines[i].set_spine_words(spine_words)
        
        print(i, spines[i].sentence)


    # Run the scraping pipeline for each spine
    books = []

    for spine in spines:

        #print([word.string for word in spine.words])

        book_info = {}

        # Get query
        search_query = spine.sentence

        print("\n" + search_query)

        # Get google search url from query
        search_url = scraper.get_google_search_url_from_query(search_query)
        print(search_url)

        # Get first amazon link from google search url
        amazon_url = scraper.get_amazon_url_from_google_search(search_url)
        print(amazon_url)

        # Get isbn10 from amazon_url
        if amazon_url != None:
            isbn10 = scraper.get_isbn10_from_amazon_url(amazon_url, debug = True)

        else:
            # Couldn't get isbn10 from amazon link (or there was no amazon link)
            isbn10 = None

        # Check if number is an isbn10; if not, skip the rest
        if not scraper.is_isbn10(isbn10, debug = False):
            print(isbn10, "is not a right ISBN.")
            continue

        """
        # Create amazon bottlenose object
        amazon = scraper.get_amazon_object()

        # Run through all the APIs
        book_info = {}
        book_price = 0


        # Try to get info from amazon products api
        book_info, book_price = scraper.query_amazon_products_api(isbn10, amazon)
        book_info['api'] = 'amazon products'
        """

        book_info, content = scraper.query_google_books_api(isbn10, type = "isbn", debug = False)
        book_info['api'] = 'amazon and google books api'

        #print(book_info)
        if(content["totalItems"] == 0):
            book_info, content = scraper.query_google_books_api(search_query, type = "title", debug = False)
            book_info['api'] = 'google books api'

        print(book_info)

        book = book_functions.Book(book_info, spine)

        books.append(book)


    # Sort books
    # Cannot sort based on the center of the book
    # books = sorted(books, key = lambda book: book.center_x)

    return books


def unpickle_all_books():
    '''
    Loads all of the books in the submissions files as a list called books
    '''

    submissions_base_path = main.BASE_PATH + '/data/submissions/'

    submissions = [dir for dir in os.listdir(submissions_base_path)]

    books_paths = [submissions_base_path + submission + '/books/' for submission in submissions]

    books = []
    for i in range(len(books_paths)):

        for file_path in os.listdir(books_paths[i]):
            with open(books_paths[i] + file_path, 'rb') as file_handle:
                books.append(pickle.load(file_handle))

    return books