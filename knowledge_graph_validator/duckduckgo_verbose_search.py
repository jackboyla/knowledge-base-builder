#!/bin/env python3

"""Simplified from https://github.com/rtwfroody/gpt-search/blob/master/gpt_search.py"""

import datetime
import time
import json
import re
from typing import Union, List, Dict

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from markdownify import MarkdownConverter
import requests
from functools import lru_cache
import chardet
import utils
import sys   
sys.setrecursionlimit(10000)

logger = utils.create_logger(__name__)

LRU_CACHE_MAXSIZE = 32


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
    try:
        return soup.title.string
    except:
        return None

class DuckDuckGoVerboseSearch:
    def __init__(self, verbose=False, max_search_results=5, max_retries=2, retry_delay=2):
        self.verbose = verbose
        self.max_search_results = max_search_results
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}


    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def fetch(self, url):
        """Fetch a URL, caching the result."""
        if self.verbose:
            logger.info(f"Fetching {url}")
        for attempt in range(1, self.max_retries + 1):
            try:
                with requests.get(url, headers=self.headers, timeout=10) as response:
                    if response.status_code == 200:
                        encoding = chardet.detect(response.content)['encoding']
                        response.encoding = encoding
                        return response.content
                    else:
                        logger.warning(f"Attempt {attempt}: Error fetching {url}: {response.status_code}")
            except Exception as exception:
                logger.warning(f"Attempt {attempt}: Error fetching {url}: {exception}")
            
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        return None

    @lru_cache(maxsize=LRU_CACHE_MAXSIZE)
    def ddg_search(self, topic):
        """Search DuckDuckGo for a topic, caching the result."""
        if self.verbose:
            logger.info(f"Search DDG for: {topic}")
        result = []
        for attempt in range(1, self.max_retries + 1):
            try:
                with DDGS() as ddgs:
                    result = ddgs.text(topic, max_results=self.max_search_results)
                    if result:
                        return [r for r in result]
            except Exception as exception: 
                logger.warning(f"Attempt : Error searching DDG for {topic}: {exception}")  # {attempt}

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        return result

    def ddg_top_hits(self, results, skip=()):
        """
            Search DuckDuckGo for a topic, and find the top hits. 
            Enter each html and return the parsed pages
        """
        
        fleshed_out_results = []
        for result in results:
            title, href, body = result['title'], result['href'], result['body'] # defaults
            if href in skip:
                continue
            if self.verbose:
                logger.info(f"Fetching {href}")
            html = self.fetch(href)
            try:
                if html:
                    extracted_title = extract_title(html)
                    if extracted_title:
                        title = str(extracted_title)
                    content = simplify_html(html)
                    if content:
                        body = content
            except Exception as e:
                logger.warning(f"Error parsing result from link {href} due to {e}")

            fleshed_out_results.append({'title': title, 'href': href, 'body': body})
        return fleshed_out_results



    def __call__(self, query) -> List[Dict]:

        results = self.ddg_search(query)
        verbose_results = self.ddg_top_hits(results,
                                            # skip=[source for source, _ in sources]
                                            )

        return verbose_results

