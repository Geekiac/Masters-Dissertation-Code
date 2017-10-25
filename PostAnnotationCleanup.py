"""After the manual annotation, cleans up the xml data
due to issues found whilst annotating the data.

This is where a single sentence has been split into multiple
sentences due to abbreviations or numbered lists containing
full stops.  These sentences are joined back into a single
sentence.
"""

#__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from AcquireData import get_xml
from Logging import LoggingWrapper
from lxml import etree
import re

def main():
    """This is the root function called by the script

    Finds where a single sentence has been split into multiple
    sentences due to abbreviations or numbered lists containing
    full stops.  These sentences are joined back into a single
    sentence.
    """
    with LoggingWrapper('PostAnnotationCleanup'):
        xml_conclusions = get_xml('conclusions_with_fw_15_lines_or_less.xml')

    re_digit_or_roman_numeral = \
        re.compile('^([0-9]+|i|ii|iii|iv|v|vi|vii|viii|xi|xii|xiii|xiv|'
                   'xv|xvi|xvii|xviii|xix|xx)(\.)?')

    new_docs = etree.Element('docs')
    for doc in xml_conclusions.find_all("doc"):
        phrase_list, is_fw, l = [], False, 0

        new_doc = etree.SubElement(new_docs, 'doc',
                                   attrib=dict(id=doc['id'], key=doc['key']))
        for sentence in doc.find_all("sentence"):
            is_fw |= sentence['is-fw'].lower() == 'true'
            text = sentence.text
            phrase_list.append(text)
            has_sentence_ending = any(map(text.endswith, ['.', '?', '!']))
            last_sentence = doc['num-sentences'] == sentence['id']
            should_add_sentence = last_sentence

            if len(text) < 8:
                pass
            elif re_digit_or_roman_numeral.match(text):
                pass
            elif any(map(text.endswith, ['i.e.', 'e.g.', 'et al.', 'w.r.t.',
                                         'cf.', 'pp.', 'etc.', 'i.i.d.'])):
                pass
            elif has_sentence_ending:
                should_add_sentence = True

            if should_add_sentence:
                if last_sentence and re_digit_or_roman_numeral.match(text):
                    pass
                else:
                    new_sentence_text = ' '.join(phrase_list)
                    if not has_sentence_ending:
                        new_sentence_text += '.'

                    if last_sentence and len(new_sentence_text) < 8:
                        pass
                    else:
                        l += 1
                        new_sentence = etree.SubElement(
                            new_doc, 'sentence', attrib={'id': str(l),
                                                         'is-fw': str(is_fw)})
                        new_sentence.text = new_sentence_text
                        phrase_list, is_fw = [], False

        new_doc.attrib['num-sentences'] = str(l)

    with open('./conclusions_with_fw_15_lines_or_less_post_cleanup.xml',
              'wb') as fp:
        fp.write(etree.tostring(new_docs, pretty_print=True))


if __name__ == '__main__':
    main()