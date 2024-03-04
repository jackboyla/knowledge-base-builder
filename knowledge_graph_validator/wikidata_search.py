import requests
import copy
import json
import urllib

from wikidata.client import Client


# TODO: can be improved to filter uninteresting properties like https://github.com/langchain-ai/langchain/blob/8af4425abd3cbe890a365cec6eac9c0ba69ee282/libs/community/langchain_community/utilities/wikidata.py#L92


def fetch_wikipedia_page_content(wikidata_id):
    """
    Fetch the entire content of a Wikipedia page for a given Wikidata ID.

    Parameters:
    - wikidata_id: The Wikidata ID of the entity.

    Returns:
    - The content of the Wikipedia page as a string.
    """
    # Fetch the Wikipedia page title for the Wikidata ID
    params_wikidata = {
        'action': 'wbgetentities',
        'ids': wikidata_id,
        'props': 'sitelinks',
        'sitefilter': 'enwiki',
        'format': 'json'
    }
    response_wikidata = requests.get("https://www.wikidata.org/w/api.php", params=params_wikidata)
    response_json_wikidata = response_wikidata.json()
    sitelinks = response_json_wikidata['entities'][wikidata_id].get('sitelinks', {})
    enwiki_title = sitelinks.get('enwiki', {}).get('title', '')
    
    if not enwiki_title:
        return "Wikipedia page title not found for the given Wikidata ID."
    
    # Fetch the content of the Wikipedia page using the title
    params_wikipedia = {
        'action': 'query',
        'format': 'json',
        'titles': enwiki_title,
        'prop': 'extracts',
        'explaintext': True,  # Return plain text content for the entire page
    }
    response_wikipedia = requests.get("https://en.wikipedia.org/w/api.php", params=params_wikipedia)
    response_json_wikipedia = response_wikipedia.json()
    page = next(iter(response_json_wikipedia['query']['pages'].values()))
    content = page.get('extract', '')
    
    return content


def fetch_labels(entity_ids):
    """
    Fetch labels for a given list of Wikidata entity IDs.

    Parameters:
    - entity_ids: List of Wikidata entity IDs.

    Returns:
    - A dictionary mapping each Wikidata entity ID to its label.
    """
    labels = {}
    # Wikidata API may have limits on the number of IDs per request; adjust if necessary.
    for i in range(0, len(entity_ids), 50):
        batch_ids = entity_ids[i:i+50]
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(batch_ids),
            'props': 'labels',
            'languages': 'en',
            'format': 'json'
        }
        response = requests.get("https://www.wikidata.org/w/api.php", params=params)
        response_json = response.json()

        for entity_id, entity in response_json['entities'].items():
            label = entity.get('labels', {}).get('en', {}).get('value', entity_id)
            labels[entity_id] = label
    return labels

def format_value(value, labels):
    """
    Convert a Wikidata value to a string representation, using labels for entity IDs.
    """
    if isinstance(value, dict):
        # For entity IDs, replace with label if available
        if 'id' in value:
            return labels.get(value['id'], value['id'])
        elif 'time' in value:
            return value['time']
        elif 'amount' in value:
            return value['amount']
        else:
            return str(value)
    else:
        return str(value)

def get_all_properties_with_labels(wikidata_id):
    """
    Fetch all properties and their values for a given Wikidata ID, 
    including labels for properties and their entity values, encapsulated by the entity name.

    Parameters:
    - wikidata_id: The Wikidata ID of the entity.

    Returns:
    - A dictionary with the entity label as the key and the properties dictionary as the value.
    """
    # Initial API call to get all claims/properties for the entity
    params = {
        'action': 'wbgetentities',
        'ids': wikidata_id,
        'props': 'claims|labels',
        'languages': 'en',
        'format': 'json'
    }
    response = requests.get("https://www.wikidata.org/w/api.php", params=params)
    entity_data = response.json()['entities'][wikidata_id]

    # Fetch the label for the entity
    entity_label = entity_data.get('labels', {}).get('en', {}).get('value', wikidata_id)

    claims = entity_data['claims']

    # Collect property IDs and value entity IDs for label fetching
    prop_ids = list(claims.keys())
    value_entity_ids = set()
    for prop_id in prop_ids:
        for claim in claims[prop_id]:
            if 'datavalue' in claim['mainsnak']:
                data_value = claim['mainsnak']['datavalue'].get('value')
                if isinstance(data_value, dict) and 'id' in data_value:
                    value_entity_ids.add(data_value['id'])

    # Fetch labels for all property IDs and value entity IDs
    all_labels = fetch_labels(list(set(prop_ids) | value_entity_ids))

    # Construct the result dictionary with labels
    properties_result = {}
    for prop_id in prop_ids:
        prop_label = all_labels.get(prop_id, prop_id)
        properties_result[prop_label] = []
        for claim in claims[prop_id]:
            if 'datavalue' in claim['mainsnak']:
                data_value = claim['mainsnak']['datavalue'].get('value')
                formatted_value = format_value(data_value, all_labels)
                properties_result[prop_label].append(formatted_value)

    # Encapsulate the result using the entity label
    encapsulated_result = {
        "entity_label": entity_label,
        "properties" : properties_result
        }

    return encapsulated_result



# https://github.com/quantexa-internal/aylien-model-entities/blob/master/research/demo-kb-explorer/aylien_kb_explorer/wikidata.py

'''
https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles=ARTICLE_NAME

As well as the accepted answer you can also use the Wikidata API directly passing in a site and a title.

The docs are at https://www.wikidata.org/w/api.php?action=help&modules=wbgetentities

Get the entity for http://en.wikipedia.org/wiki/Karachi:

If you know the exact title: https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=Karachi
With title normalization, for initial character capitalization fixes, and underscores (and possible more): 
https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=Karachi&normalize=1


'''


class WikidataSearchError(Exception):
    pass



class WikidataSearch:
    """
    This is a "KB Searcher" implemetation for Wikidata
    """
    WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
    DEFAULT_SEARCH_PARAMS = {
        "action": "wbsearchentities",
        "format": "json",
        "errorformat": "plaintext",
        "language": "en",
        "uselang": "en",
        "type": "item",
        "limit": 1
    }
    WD_ENTITY_BASE_URL = 'https://www.wikidata.org/wiki/Special:EntityData'

    def __call__(self,  q):
        return self.best_wikidata_entities_from_surface_form(q)

    @staticmethod
    def extract_aliases(wd_json, lang='en'):
        aliases = []
        if lang in wd_json.get('aliases', {}):
            aliases = [a['value'] for a in wd_json['aliases'][lang]]
        return aliases

    @staticmethod
    def extract_description(wd_json, lang='en'):
        try:
            return wd_json['descriptions'][lang]['value']
        except KeyError as e:
            print(f'Error extracting wikidata description: {e}')
            print(f'full json: {wd_json}')
            return ''

    @staticmethod
    def extract_label(wd_json, lang='en'):
        try:
            return wd_json['labels'][lang]['value']
        except KeyError as e:
            print(f'Error extracting wikidata description: {e}')
            print(f'full json: {wd_json}')
            return ''

    @classmethod
    def search_wikidata(cls, surface_form, min_length=3):
        if len(surface_form) < min_length:
            print(f'query surface form: {surface_form} is too short')
            return []

        params = copy.deepcopy(cls.DEFAULT_SEARCH_PARAMS)
        params["search"] = surface_form
        result = []
        try:
            print(f'querying wikidata with params: {params}')
            r = requests.get(url=cls.WIKIDATA_SEARCH_URL, params=params)
            data = json.loads(r.text)
            if 'search' in data:
                result = data['search']
                result = result[0:min(100, len(result))]
        except Exception as e:
            print(f'Error searching wikidata for surface form: {surface_form}')
            print(e)
        return result

    @classmethod
    def wikidata_id_from_wikipedia_pagename(cls, page_name):
        params = {
            "action": "wbgetentities",
            "format": "json",
            "sites": 'enwiki',
            "errorformat": "plaintext",
            "normalize": 1,
            "titles": page_name
        }
        try:
            r = requests.get(
                url=cls.WIKIDATA_SEARCH_URL, params=params
            )
            return [e_data for e, e_data in r.json()['entities'].items()]
        except Exception as e:
            print(
                f'Error retrieving wikidata item from wikipedia page title: {e}'
            )
            print(f'params: {params}')
            return []

    @staticmethod
    def aylien_kb_entity_from_wikidata_id(
        wikidata_id,
        user_id='kb-explorer-ui',
        status='pending'
    ):
        """
        Init an Aylien entity from a wikidata id
        :param wikidata_id:
        :return:
        """
        client = Client()
        try:
            entity = client.get(wikidata_id, load=True)
        except urllib.error.HTTPError as e:
            print(f'Error retrieving wikidata entity: {wikidata_id}')
            raise WikidataSearchError(e.reason)

        # TODO: support adding supported types from Wikidata
        # TODO: support adding tickers from Wikidata
        return db.Entity(
            entity_id=str(entity.id),
            wikiname=str(entity.label),
            description=str(entity.description),
            user_entities=[
                db.UserEntity(
                    user_id=user_id,
                    entity_id=str(entity.id),
                    status=status
                )
            ],
            # types=new_entity_types,
            stock_tickers=[],
            wd_description=str(entity.description),
            wd_label=str(entity.label),
            created_at=None,
            updated_at=None
        )

    @staticmethod
    def wikipedia_link_from_wikidata_id(wikidata_id):
        """
        Try to return the English wikipedia page for a wikidata item
        :param wikidata_id:
        :return:
        """
        client = Client()

        url = None
        try:
            entity = client.get(wikidata_id, load=True)
            url = entity.data['sitelinks']['enwiki']['url']
        except KeyError:
            print(f'Error: no wikipedia url found for entity data: {entity.data}')
        except urllib.error.HTTPError as e:
            print(f'Error retrieving wikidata entity: {wikidata_id}')

        return url

    @classmethod
    def wd_entity_json(cls, wd_id):
        json_url = f'{cls.WD_ENTITY_BASE_URL}/{wd_id}.json'
        resp = None
        try:
            r = requests.get(url=json_url)
            resp = r.json()['entities'][wd_id]
        except Exception as e:
            print(f"Error {e} retrieving json data for wd id: {wd_id}")
        return resp

    @classmethod
    def best_wikidata_entities_from_surface_form(cls, sf):
        srs = cls.search_wikidata(sf)
        matches = []
        if len(srs):
            for sr in srs:
                sr_json = cls.wd_entity_json(sr['id'])
                matches.append({
                    'wikidata_id': sr['id'],
                    'label': cls.extract_label(sr_json),
                    'description': cls.extract_description(sr_json),
                    'aliases': cls.extract_aliases(sr_json),
                    'types': []
                })
        return matches
