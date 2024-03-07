#!/bin/env python3

"""Simplified from https://github.com/rtwfroody/gpt-search/blob/master/gpt_search.py"""

import datetime
import json
import re
import sys
from typing import Union, List, Dict

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from markdownify import MarkdownConverter
import requests
from functools import lru_cache

LRU_CACHE_MAXSIZE = 128


def simplify_html(html):
    """Convert HTML to markdown, removing some tags and links."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Remove links. They're not helpful.
    for tag in soup.find_all("a"):
        del tag["href"]
    for tag in soup.find_all("img"):
        del tag["src"]
    soup.smooth()

    # Turn HTML into markdown, which is concise but will attempt to
    # preserve at least some formatting
    text = MarkdownConverter().convert_soup(soup)
    text = re.sub(r"\n(\s*\n)+", "\n\n", text)
    return text

def extract_title(html):
    """Extract the title from an HTML document."""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.title.string

class DuckDuckGoVerboseSearch:
    def __init__(self, verbose=False, max_search_results=5):
        self.verbose = verbose
        self.max_search_results = max_search_results

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def fetch(self, url):
        """Fetch a URL, caching the result."""
        if self.verbose:
            print("Fetching", url)
        try:
            # Fetch the URL
            response = requests.get(url, timeout=10)
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error fetching {url}: {response.status_code}")
                return None
            return response.content
        except requests.RequestException as exception:
            print(f"Error fetching {url}: {exception}")
            return None

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def ddg_search(self, topic):
        """Search DuckDuckGo for a topic, caching the result."""
        if self.verbose:
            print("Search DDG for:", topic)
        result = DDGS().text(topic, max_results=self.max_search_results)  # a generator of dict['title', 'href', 'body']
        return result

    def ddg_top_hits(self, topic, skip=()):
        """
            Search DuckDuckGo for a topic, and find the top hits. 
            Enter each html and return the parsed pages
        """
        results = self.ddg_search(topic)
        fleshed_out_results = []
        for result in results:
            title, href, body = result['title'], result['href'], result['body'] # defaults
            if href in skip:
                continue
            if self.verbose:
                print("  Fetching", href)
            html = self.fetch(href)
            if html:
                extracted_title = extract_title(html)
                if extracted_title:
                    title = str(extracted_title)
                content = simplify_html(html)
                if content:
                    body = content

            fleshed_out_results.append({'title': title, 'href': href, 'body': body})
        return fleshed_out_results



    def __call__(self, query) -> List[Dict]:

        verbose_results = self.ddg_top_hits(query,
                                            # skip=[source for source, _ in sources]
                                            )

        return verbose_results

