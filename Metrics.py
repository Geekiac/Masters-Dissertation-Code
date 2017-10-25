"""Monitors the progress of the manual annotation of the
conclusions_with_fw_15_lines_or_less.xml file
"""

__all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from AcquireData import get_xml
from Logging import log, LoggingWrapper


def main():
    """The root function to execute when the script executes.
    """
    xml_file_to_process = 'conclusions_with_fw_15_lines_or_less.xml'
    with LoggingWrapper("Loading XML file: %s" % xml_file_to_process):
        soup = get_xml(xml_file_to_process)

    notProcessedDocs = [doc
                        for doc in soup.find_all('doc')
                        if doc["processed"] != "True"]
    len_notProcessedDocs = len(notProcessedDocs)
    log("Num docs still to process = %d" % len_notProcessedDocs)

    notProcessedSentences = [s
                             for doc in notProcessedDocs
                             for s in doc.find_all('sentence')]
    # sentences plus doc start nodes and doc terminating nodes
    log("Num lines still to process = %d ****"
        % (len(notProcessedSentences) + (2*len_notProcessedDocs)))

    processedDocs = [doc
                     for doc in soup.find_all('doc')
                     if doc["processed"] == "True"
                     and doc.get("ignore") != "True"]
    log("Num processed docs = %d" % len(processedDocs))

    allProcessedSentence = [s
                            for doc in processedDocs
                            for s in doc.find_all('sentence')]
    log("Total processed sentences = %d" % len(allProcessedSentence))

    fwDocs = [doc
              for doc in soup.find_all('doc')
              if doc["processed"] == "True"
              and doc.get("ignore") != "True"
              and [s
                   for s in doc.find_all('sentence')
                   if s['is-fw'] == 'True']]
    len_fwDocs = len(fwDocs)
    log("Num FW docs = %d" % len_fwDocs)

    fwSentence = [s
                  for doc in fwDocs
                  for s in doc.find_all('sentence')
                  if s['is-fw']=='True']
    log("Num FW sentences = %d" % len(fwSentence))

    allSentences_FW = [s
                       for doc in fwDocs
                       for s in doc.find_all('sentence')]
    log("Total processed sentences in fwDocs = %d" % len(allSentences_FW))


    log("")
    log("Possible training/test splits")
    log("=============================")

    for training_split in [90, 80, 70]:
        num_training = (training_split/100) * len_fwDocs
        num_test = (1-(training_split/100)) * len_fwDocs
        log("%d/%d would give %d/%d" % (training_split, 100 - training_split,
                                        num_training, num_test))


if __name__ == '__main__':
    main()
