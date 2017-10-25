"""Calls the CERMINE jar file repeatedly, recovering from errors,
until there are no more files to process.
"""

# __all__ = []
__version__ = '0.1'
__author__ = "Steven Smith"

from Logging import LoggingWrapper, log
import os
import subprocess


def main():
    """The root function called by the script.
    Calls the CERMINE jar file repeatedly, recovering from errors,
    until there are no more files to process.
    """
    pdf_path = './pdf'

    with LoggingWrapper("ConvertPdfToXmlFiles"):
        while True:
            with LoggingWrapper("Executing CERMINE"):
                command = 'java -cp ' \
                          'cermine-impl-1.13-jar-with-dependencies.jar ' \
                          'pl.edu.icm.cermine.ContentExtractor' \
                          ' -path "%s" -outputs "zones" -exts "xml"' \
                          % pdf_path
                result = subprocess.run(command.split(),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        check=False)
                output = result.stdout.decode('utf-8').split(os.linesep)
                files = [l.replace('File processed: ', '')
                         for l in output
                         if l.startswith('File processed: ')]
                files_processed = len(files)
                log("Number of files processed = %s" % files_processed)

            if files_processed == 0:  # nothing left to process
                break

            # Gets the last file processed and checks if there was a problem.
            # If there is a problem creates an XML stub so the PDF will not
            # be processed again.
            f = files[-1]
            xml_file = f.replace('.pdf', '.xml')
            if not os.path.exists(xml_file):
                log("Could not convert %s" % f)
                with open(xml_file, 'w', encoding='utf8') as fp:
                    fp.write("<document></document>\n")

if __name__ == "__main__":
    main()
