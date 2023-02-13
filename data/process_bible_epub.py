import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

book = epub.read_epub('bible_nt.epub', options={"encoding": "utf-8"})
items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
print(len(items))


def chapter_to_str(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = soup.text.replace('\n', ' ').strip()
    return text


text = ""
for item in items[1:]:
    text += chapter_to_str(item)

print(len(text), "\n", text[:1000])

## data cleaning
# @WIP
