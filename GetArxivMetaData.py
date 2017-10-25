"""Retrieves the arXiv meta-data and stores it in
search_results_000.xml
"""

# __all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from AcquireData import download_url_to_file
from Logging import LoggingWrapper, log
from os import path
from argparse import ArgumentParser


def process_arguments():
    """Processes the arguments from the command line

    :rtype: argparse.Namespace
    """
    parser = ArgumentParser(
        description='Retrieves the meta data from '
                    'http://export.arxiv.org and saves as an xml file.')
    parser.add_argument('-f', '--force', action='store_true',
                        default=False, help='force overwrite of xml file')
    parser.add_argument('-n', '--numpdfs', type=int, default=10000,
                        help='number of pdfs to retrieve - default 10')
    parser.add_argument('-x', '--xmlfilename', type=str,
                        default='search_results_000.xml',
                        help='xml filename to save meta-data to '
                             '- default search_results_000.xml')
    return parser.parse_args()


def main():
    """The root function called by the script.
    Retrieves the arXiv meta-data and stores it in a file.
    """
    args = process_arguments()

    force_overwrite = args.force
    max_num_pdfs = args.numpdfs
    metadata_xml_filename = args.xmlfilename

    with LoggingWrapper("GetArxivMetaData"):
        if force_overwrite or not path.exists(metadata_xml_filename):
            url = 'http://export.arxiv.org/api/query?' \
                  'search_query=cat:cs.AI&start=0&' \
                  'max_results=%d&sortBy=submittedDate&sortOrder=ascending' \
                  % max_num_pdfs

            with LoggingWrapper('Downloading metadata [%d items]'
                                % max_num_pdfs):
                download_url_to_file(url, metadata_xml_filename,
                                     force_overwrite)

        if path.exists(metadata_xml_filename):
            log('%s file exists!' % metadata_xml_filename)

if __name__ == "__main__":
    main()




