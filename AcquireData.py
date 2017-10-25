"""Functions to acquire data by:

1) downloading files
2) parsing xml
"""

__all__ = ['download_url', 'download_url_to_file', 'get_xml']
__version__ = '0.1'
__author__ = "Steven Smith"

from bs4 import BeautifulSoup
import os
import requests


def download_url(url):
    """Downloads a url and returns a Response object.
    NOTE: The arXiv API requires a user-agent or rejects the request.

    :param url: url address to retrieve
    :return: a Response object
    """
    return requests.get(url, headers={'user-agent': 'Lynx'})


def download_url_to_file(url, filepath, overwrite_file=True):
    """Download data from a url and save to file.

    :param url: url address to retrieve
    :param filepath: file path to save the data to
    :param overwrite_file: if a file exists should it be overwritten?
    :return: a Response object
    """
    if overwrite_file or not os.path.exists(filepath):
        r = download_url(url)
        with open(filepath, 'wb') as f:
            f.write(r.content)
    return r


def get_xml(xml_filename):
    """Parse xml from a file.

    :param xml_filename: the xml filename to parse
    :return: a BeautifulSoup xml object
    """
    with open(xml_filename, encoding='utf8') as fp:
        return BeautifulSoup(fp, 'lxml-xml')
