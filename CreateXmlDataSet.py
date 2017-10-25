"""Calls the CERMINE jar file repeatedly, recovering from errors,
until there are no more files to process.
"""

# __all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from multiprocessing.pool import Pool
from functools import partial
from lxml import etree
import re
import pandas as pd
import os
from nltk import sent_tokenize
from AcquireData import get_xml
from CleanUpData import is_valid_conclusion_or_fw_heading, clean_up_heading, \
                        repair_sentence
from Logging import LoggingWrapper, log


def get_element_text_or_default(element, name, default=''):
    """Gets the text from an under a root element/node or returns
    the default value.

    :param element: the root element to get the text from
    :param name: the name of the element to look for
    :param default: the default value
    :return: element text or the default value
    """
    e = element.find(name)
    return e.text if e else default


def get_all_terms_for_category(entry, category_name):
    """Creates a comma separated list of all the category names

    :param entry: the entry to get the category name list from
    :param category_name: the type of category to obtain
    :return: a comma separated list of categories as a string
    """
    return ', '.join(c['term'] for c in entry.find_all(category_name) if c)


def get_sentences_from_zone(zone, found_heading):
    """Processes a zone to obtain the sentences

    :param zone: the zone to retrieve sentences from
    :param found_heading: whether the conclusion heading has been found
    :return: a tuple of three items: a list of sentences,
                                     if conclusion heading was found,
                                     if processing should finish
    """
    finished_processing = False
    sentences = []
    if zone.attrib['label'] in ["BODY_HEADING"]:
        if not found_heading:
            if is_valid_conclusion_or_fw_heading(zone.text):
                found_heading = True
        else:
            clean_heading = clean_up_heading(zone.text)
            if any(map(clean_heading.lower().startswith, ['appendi',
                                                          'refer',
                                                          'acknowledg'])):
                finished_processing = True
    elif zone.attrib['label'] in ["GEN_REFERENCES"]:
        references = [reference
                      for reference in zone.text.split('\n')
                      if reference and not reference.isspace()]
        if references and references[0].lower().startswith('refer'):
            finished_processing = True
    elif zone.attrib['label'] in ["BODY_CONTENT", "MET_ABSTRACT"]:
        if found_heading:
            clean_sentences = [repair_sentence(sentence)
                               for sentence in sent_tokenize(
                                                zone.text.replace('\n', ' '))]

            terminating_section_found = False
            for sentence in clean_sentences:
                if any(map(sentence.lower().startswith, ['appendi',
                                                         'refer',
                                                         'acknowledg'])):
                    terminating_section_found = True
                    break
                sentences.append(sentence)

            if terminating_section_found:
                finished_processing = True

    return sentences, found_heading, finished_processing


def get_conclusion_xml(zones, key):
    """ Create conclusion xml from a list of xml zones.

    :param zones: list of xml zones
    :param key: the key of the document
    :return: returns the conclusion xml for the current document
    """
    found_heading = False
    sentences = []
    for zone in zones:
        new_sentences, found_heading, finished_processing = \
            get_sentences_from_zone(zone, found_heading)
        sentences.extend(new_sentences)
        if finished_processing:
            break

    doc = etree.Element('doc', {'key': key, 'processed': 'False'})
    l = 0
    for s in sentences:
        l += 1
        sentence = etree.Element("sentence", {'id': str(l), 'is-fw': 'False'})
        sentence.text = s
        doc.append(sentence)

    doc.attrib["num-sentences"] = str(l)
    #print(key, l)
    return doc, l


def path_join_and_exists(path, *paths):
    """Joins paths together and checks the path exists

    :param path: the root path
    :param paths: the paths to append
    :return: True if the joined path exists otherwise False
    """
    joined_paths = os.path.join(path, *paths)
    return os.path.exists(joined_paths)


def update_entry_with_match(entry, pdf_path, load_xml, match):
    """ Refresh the entry with additional information

    :param entry: the entry dictionary to update
    :param pdf_path: the path to the pdfs
    :param load_xml: whether to load the pdf xml file
    :param match: the regular expression matches breaking up the identifier
    """
    entry['root_category'] = match.groups(0)[0]
    entry['key'] = match.groups(0)[1]
    entry['version'] = match.groups(0)[2]
    entry['new_key_format'] = entry['key'].find('.') > -1
    if entry['new_key_format']:  # New filename format contains .
        entry['filename'] = entry['key']
    else:
        entry['filename'] = '%s%s' % (entry['root_category'],
                                      entry['key'])

    entry['pdf_exists'] = \
        path_join_and_exists(pdf_path, '%s.pdf' % entry['filename'])
    entry['xml_exists'] = \
        path_join_and_exists(pdf_path, '%s.xml' % entry['filename'])
    entry['html_exists'] = \
        path_join_and_exists(pdf_path,'%s.missing.html' % entry['filename'])
    entry['zones_exist'] = False
    if load_xml and entry['xml_exists']:
        full_xml = etree.parse(
            os.path.join(pdf_path, '%s.xml' % entry['filename']))
        zones = full_xml.findall('zone')
        entry['zones_exist'] = len(zones) > 0
        conclusion_xml, num_sentences = get_conclusion_xml(zones,
                                                           entry['key'])
        entry['conclusion_xml'] = etree.tostring(conclusion_xml)
        entry['num_sentences'] = num_sentences


def update_dictionary_entry(reg_exps, pdf_path, load_xml, entry):
    """Updates the entry dictionary once the correct regular expression has
    matched with the identifier to determine if it is the old or new format.

    :param reg_exps: regular expressions for the old and new format identifiers
    :param pdf_path: the path to the pdf files
    :param load_xml: whether to load the pdf xml file
    :param entry: the entry dictionary to update
    :return: the entry dictionary
    """
    try:
        for reg_exp in reg_exps:
            match = reg_exp.match(entry['id'])
            if match:
                update_entry_with_match(entry, pdf_path, load_xml, match)
                break
    except:
        print(entry)
        raise

    return entry


def entry_from_xml(xml):
    """Creates the initial entry from the xml

    :param xml: the xml containing initial entry information
    :return: dictionary contain the initial entry
    """
    return dict(id=(xml.id and xml.id.text) or '',
                key='',
                version='',
                filename='',
                title=get_element_text_or_default(xml, 'title'),
                published=get_element_text_or_default(xml, 'published'),
                comment=get_element_text_or_default(xml, 'comment'),
                journal_ref=get_element_text_or_default(xml, 'journal_ref'),
                authors=', '.join(a.text.strip()
                                  for a in xml.find_all('author')),
                category=get_all_terms_for_category(xml, 'category'),
                primary_category=
                    get_all_terms_for_category(xml, 'primary_category'),
                root_category='',
                pdf_exists=False,
                xml_exists=False,
                html_exists=False,
                zones_exist=False,
                conclusion_xml=None,
                num_sentences=0)


def update_dictionary_entry_with_reg_exps(pdf_path, load_xml):
    """Get a partially initialised function to update the entry dictionary.

    :param pdf_path: the path to the pdfs
    :param load_xml: whether to load the pdf xml file
    :return: a function to update the entry dictionary partially initialised
    """
    # old style: http://arxiv.org/abs/cs/9311102v1
    # new style: http://arxiv.org/abs/1005.1684v12
    reg_exps = [re.compile('.*/(.*?)/([0-9]+)v([0-9]+)$'),
                re.compile('.*/(.*?)/([0-9]+?\.[0-9]+?)v([0-9]+)$')]
    return partial(update_dictionary_entry, reg_exps, pdf_path, load_xml)


def create_data_frame(xml_papers, pdf_path, load_xml=True):
    """Creates a pandas DataFrame containing the papers meta data.

    :param xml_papers: the xml meta data for the research papers
    :param pdf_path: the path to the pdfs
    :param load_xml: whether to load the pdf xml file
    :return: a pandas DataFrame containing the papers meta data.
    """
    all_entry_xml = [entry_from_xml(entry)
                     for entry in xml_papers.find_all('entry')]

    # allows for distribution of execution on multiple cpus
    with Pool() as pool:
        entries = [e for e in pool.map(
            update_dictionary_entry_with_reg_exps(pdf_path, load_xml),
            all_entry_xml)]
    return pd.DataFrame(entries)


def write_conclusion_xml_file(papers_df):
    """Outputs the conclusion xml to file.

    :param papers_df: papers meta data DataFrame
    """
    papers_with_conclusion = papers_df[(papers_df["num_sentences"] >= 1)
                                       & (papers_df["num_sentences"] <= 100)]
    print("len(papers_df) =", len(papers_df))
    print("len(papers_with_conclusion) =", len(papers_with_conclusion))
    papers_with_conclusion = \
        papers_with_conclusion.sort_values('num_sentences')
    docs = etree.Element("docs")
    doc_id = 0
    for row in papers_with_conclusion.itertuples():
        doc_id += 1
        doc = etree.XML(row.conclusion_xml)
        doc.attrib['id'] = str(doc_id)
        docs.append(doc)

    xml_output_filename = './conclusions_with_fw.xml'
    with open(xml_output_filename, mode='wb') as f:
        f.write(etree.tostring(docs, pretty_print=True))

    log("Created %d documents in the XML file." % doc_id)


def main():
    """The root function executed when calling the script.
    """
    with LoggingWrapper('CreateXmlDataSet'):
        with LoggingWrapper('Getting xml papers'):
            xml_papers = get_xml("./search_results_000.xml")

        with LoggingWrapper("Created initial data frame"):
            papers_df = create_data_frame(xml_papers, pdf_path="./pdf/")

        with LoggingWrapper("Write conclusion xml file"):
            write_conclusion_xml_file(papers_df)


if __name__ == '__main__':
    main()
