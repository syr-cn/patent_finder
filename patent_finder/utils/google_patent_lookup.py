import os
import sys
from serpapi import GoogleSearch
import requests
import re
from markdownify import markdownify as md
from google_patent_scraper import scraper_class

from .image_parser import download_images
import json
import time


def get_response(url, retry=3, timeout=10):
    for _ in range(retry):
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response
        time.sleep(2)
    return None

serpapi_key = None
def get_patent_api(patent_id):
    # Deprecated
    search = GoogleSearch({
        "engine": "google_patents",
        "q": patent_id,
        "api_key": serpapi_key
    })
    results = search.get_dict()
    result = results['organic_results'][0]
    serpapi_link = f"{result['serpapi_link']}&api_key={serpapi_key}"
    return get_response(serpapi_link).json()

def get_patent_scrap(patent_id):
    """
    get the claim text from Google Patent
    """
    scraper = scraper_class()
    err_1, html_content, url_1 = scraper.request_single_patent(patent_id)
    soup_claim = html_content.find('section', itemprop='claims')

    text = ''
    for div in soup_claim.find_all('div'):
        if div.get('class') and 'claim-text' in div.get('class'):
            text += md(str(div), escape_misc=False).strip() + '\n\n'
    return text

def get_fulltext_patent_scrap(patent_id):
    """
    get the full text from Google Patent.
    include: abstract, description, claims
    """
    scraper = scraper_class()
    err_1, html_content, url_1 = scraper.request_single_patent(patent_id)

    abstract_section = html_content.find('section', itemprop='abstract')
    abstact_text = md(str(abstract_section), escape_misc=False).strip()
    abstact_text = re.sub(r'\n\s*\n\s*[\n\s]*\n', '\n\n', abstact_text)

    claim_section = html_content.find('section', itemprop='claims')
    claim_text = md(str(claim_section), escape_misc=False).strip()
    claim_text = re.sub(r'\n\s*\n\s*[\n\s]*\n', '\n\n', claim_text)

    description_section = html_content.find('section', itemprop='description')
    description_text = md(str(description_section), escape_misc=False).strip()
    description_text = re.sub(r'\n\s*\n\s*[\n\s]*\n', '\n\n', description_text)

    return abstact_text, claim_text, description_text

def google_patent_scrap(cache_path, patent_id, endpoint_url):
    if os.path.exists(f'{cache_path}/full_text.txt'):
        return load_from_cache(cache_path, patent_id, endpoint_url=endpoint_url)
    print(f'Cache {cache_path} not found. Extracting {patent_id} from Google Patent.')
    os.makedirs(cache_path, exist_ok=True)
    abstact_text, claim_text, description_text = get_fulltext_patent_scrap(patent_id)
    full_text = '\n\n\n'.join([abstact_text, claim_text, description_text])
    text_dict = {
        "full": full_text,
        "abstract": abstact_text,
        "claim": claim_text,
        "description": description_text,
    }
    for key, text in text_dict.items():
        with open(f'{cache_path}/{key}_text.txt', 'w') as f:
            f.write(text)
        print(f'Patent text saved to {cache_path}/{key}_text.txt')

    for key, text in text_dict.items():
        text_dict[key], image_links = extract_image_urls(text)
        if key == 'full':
            image_dict = download_images(image_links, cache_path, endpoint_url=endpoint_url)
    return {
        "text": text_dict,
        "images": image_dict
    }

def google_patent_lookup(cache_path, patent_id):
    """
    input: patent_id: str
    output: Dict: {
        "text": str,
        "claims": List[Dict]
    }
    """
    # Deprecated
    raise NotImplementedError
    if os.path.exists(f'{cache_path}/claim_text.txt'):
        return load_from_cache(cache_path, patent_id)
    print(f'Cache {cache_path} not found. Extracting {patent_id} from SerpAPI.')
    os.makedirs(cache_path, exist_ok=True)
    patent_info = get_patent_api(patent_id)
    if patent_info is None:
        print(f'Failed to get patent info for {patent_id}.')
        return None
    description_link = patent_info['description_link']
    print(f'Extracting from {description_link}')
    description_html = get_response(description_link).text
    description_md = md(description_html, escape_misc=False)
    with open(f'{cache_path}/{patent_id}.md', 'w') as f:
        print(f'Extraction success. saving to {cache_path}/{patent_id}.md')
        f.write(description_md)
    
    claim_text = '\n'.join(patent_info['claims'])
    with open(f'{cache_path}/claim_text.txt', 'w') as f:
        f.write(claim_text)

    description_md, image_links = extract_image_urls(description_md)
    image_dict = download_images(image_links, cache_path)
    sys.stdout.flush()
    return {
        "text": claim_text,
        "images": image_dict
    }

def extract_image_urls(text):
    pattern = r'\[\!\[.*?\]\((https://patentimages\.storage\.googleapis\.com/[^"]+?.png)\)\].*?\n'
    text = re.sub(pattern, r'\1\n', text)
    urls = re.findall(r'https://patentimages\.storage\.googleapis\.com/[^"\s]+?.png', text)

    # Remove duplicates
    seen = set()
    urls = [url for url in urls if not (url in seen or seen.add(url))]
    return text, urls

def load_from_cache(cache_path, patent_id, endpoint_url):
    text_key_list = ['full', 'abstract', 'claim', 'description']
    text_dict = {}
    for key in text_key_list:
        text_path = f'{cache_path}/{key}_text.txt'
        with open(text_path, 'r') as f:
            text_dict[key] = f.read()

    image_map = {}
    for file_name in os.listdir(cache_path):
        if file_name.endswith('.json') and file_name.startswith('image_'):
            with open(f'{cache_path}/{file_name}', 'r') as f:
                image_dict = json.load(f)
            image_dict['path'] = f'{cache_path}/{file_name}'.replace('.json', '.png')
            image_dict['json_path'] = f'{cache_path}/{file_name}'
            image_map[image_dict['link']] = image_dict

    # Check cache integrity
    for key, text in text_dict.items():
        text_dict[key], image_links = extract_image_urls(text)
        if key == 'full':
            extracted_urls = image_links

    if os.path.exists(f'{cache_path}/url_404_list.json'):
        with open(f'{cache_path}/url_404_list.json', 'r') as f:
            links_404_list = json.load(f)
    else:
        links_404_list = []
        with open(f'{cache_path}/url_404_list.json', 'w') as f:
            json.dump(links_404_list, f, indent=4)

    for url in extracted_urls:
        if url in links_404_list:
            continue
        if url not in image_map:
            print(f'{patent_id} Image missing in cache. Re-downloading images.')
            print(f'Missing url: {url}')
            image_list = download_images(extracted_urls, cache_path, endpoint_url=endpoint_url)
            break
    else:
        image_list = [image_map[url] for url in extracted_urls if url in image_map]
    assert image_list
    print(f"Loaded from cache dir: {cache_path}. patent_id: {patent_id}")
    sys.stdout.flush()
    return {
        "text": text_dict,
        "images": image_list,
    }