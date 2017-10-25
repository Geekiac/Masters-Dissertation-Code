"""Gets the PDF files from the tar files.

The PDF tar files are expected to exist in either:

/arXiv/pdf
/media/geekiac/Storage/arXiv/pdf
c:/arXiv/pdf
d:/arXiv/pdf

If the PDF file is not in the tar file an attempt will be made to download from
the web.

If the file cannot be downloaded from the web save a file with a
".missing.html" extension to mark the file as downloaded and to prevent the
file from being requested again.
"""


# __all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from AcquireData import download_url_to_file, get_xml
from contextlib import suppress
from CreateXmlDataSet import create_data_frame
import glob
import os
from Logging import LoggingWrapper
import tarfile

tar_file_paths = ['/arXiv/pdf',
                  '/media/geekiac/Storage/arXiv/pdf',
                  'c:/arXiv/pdf',
                  'd:/arXiv/pdf']


def extract_file_from_arxiv_tar_files(tar_path, key, pdf_filename,
                                      output_path):
    """Extract a pdf file from the a tar file

    :param tar_path: the path to the tar files
    :param key: the key to look for
    :param pdf_filename: the pdf filename to find
    :param output_path: the path to output the pdf file to
    :return: True if extracted to a file otherwise False
    """
    inner_path = key[0:4]
    tar_filenames = glob.glob(os.path.join(tar_path,
                                           'arXiv_pdf_%s_*.tar' % inner_path))

    for tar_filename in tar_filenames:
        tar = tarfile.open(tar_filename)
        with suppress(KeyError):
            member = tar.getmember(os.path.join(inner_path, pdf_filename))
            # changing the internal filename to be the filename to output
            member.name = pdf_filename
            tar.extract(member, output_path)
            return True

    return False


def select_first_or_default(func, iterable_items, default=None):
    """Gets the first filtered item or returns a default if can't find a match

    :param func: the function to filter the items with
    :param iterable_items: the iterable items
    :param default: the default value if no items match the filter
    :return: the first filtered item or returns default if can't find a match
    """
    return next(filter(func, iterable_items), default)


def main():
    """The root function executed when the script is executed.

    Trys to get PDFs from tar files or the web.
    """
    with LoggingWrapper('GetPdfFiles'):
        with LoggingWrapper('Getting xml papers'):
            xml_papers = get_xml("search_results_000.xml")

        with LoggingWrapper("Created initial data frame"):
            papers_df = create_data_frame(xml_papers, pdf_path="./pdf/",
                                          load_xml=False)

        n, t, pdf_files_exist, missing_files_exist = 0, 0, 0, 0
        tar_file_path = select_first_or_default(os.path.exists,
                                                tar_file_paths)

        for row in papers_df.itertuples():
            with LoggingWrapper("Processing %s" % row.key):
                if n >= 50:
                    print("downloaded maximum of", n, "files.")
                    break

                if row.new_key_format:
                    filename = '%s.pdf' % row.key
                else:
                    filename = '%s%s.pdf' % (row.root_category, row.key)

                filepath = './pdf/%s' % filename
                if os.path.exists(filepath):
                    pdf_files_exist += 1
                    continue

                missing_filepath = filepath.replace('.pdf', '.missing.html')
                if os.path.exists(missing_filepath):
                    missing_files_exist += 1
                    continue

                # try extracting from the tar file
                if not os.path.exists(filepath) and tar_file_path:
                    extract_file_from_arxiv_tar_files(tar_file_path, row.key,
                                                      filename, './pdf')
                    if os.path.exists(filepath):
                        t += 1
                        continue

                # not all of the pdfs are in the tar files so, some need to be
                # downloaded (some are html versions only)
                if not os.path.exists(filepath):
                    n += 1
                    pdf_url = row.id.replace('/abs/', '/pdf/')
                    with LoggingWrapper("Downloading %s" % pdf_url):
                        request = download_url_to_file(pdf_url, filepath)
                        content_type = request.headers.get('Content-Type',
                                                           'text/text')

                    if content_type != 'application/pdf':
                        os.rename(filepath, missing_filepath)
                        print('Renamed pdf')

                if not os.path.exists(filepath) \
                        and not os.path.exists(missing_filepath):
                    print((row[0], filename, getattr(row, 'new_key_format'),
                           row.id, pdf_url))

        print(pdf_files_exist, "pdf_files_exist")
        print(missing_files_exist, "missing_files_exist")
        print(t, "files were in the tar file")


if __name__ == '__main__':
    main()
