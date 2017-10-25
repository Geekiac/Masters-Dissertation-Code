PLEASE NOTE
===========

This minimal zip had to be less than 40MB, so some files have been deleted.
The following files would need to be downloaded:

The CERMINE jar file can be obtained from https://github.com/CeON/CERMINE and the file should be but into the folder structure below:

./cermine-impl-1.13-jar-with-dependencies.jar

The Stanford NLP POS Tagger can be obtained from: https://nlp.stanford.edu/software/tagger.shtml and the files should be but into the folder structure below:

./stanford-postagger.jar
./models/english-bidirectional-distsim.tagger
./models/english-bidirectional-distsim.tagger.props
./models/english-left3words-distsim.tagger
./models/english-left3words-distsim.tagger.props
./models/README-Models.txt


Executable Scripts
==================
All of these scripts can be executed without parameters:

e.g. python GetArxivMetaData.py

PLEASE NOTE: Most of these files won't find work to do as the files they are
to generate have already been generated!

01. GetArxivMetaData.py
02. GetPdfFiles.py
03. ConvertPdfToXmlFiles.py
04. CreateXmlDataSet.py
05. Metrics.py
05. PostAnnotationCleanup.py
06. AnalyseSentences.py
07. NBandSVMTests.py
08. CNNTests.py
09. AnalyseNBandSVMResults.py
10. AnalyseCNNResults.py

Supporting Files containing library functions
=============================================
01. AcquireData.py
02. CleanUpData.py
03. Logging.py

arXiv meta-data and conclusions xml files
=========================================
01. search_results_000.xml
02. conclusions_with_fw.xml
03. conclusions_with_fw_15_lines_or_less.xml
04. conclusions_with_fw_15_lines_or_less_post_cleanup.xml

Serialized Pandas DataFrames
============================
01. conclusions_dataframe.pickle
02. cnn_results.pickle
03. nb_and_svm_results.pickle

CERMINE application
===================
cermine-impl-1.13-jar-with-dependencies.jar

Stanford NLP POS tagger
=======================
stanford-postagger.jar

./dataframe/ - is a backup of the DataFrames in pickle format
./pdf/ - This folder contains the PDF and XML files used in the experiments
./logs/ - Contains the TensorFlow logs generated during the CNN tests
./models/ - contains POS tagging models for stanford-postagger.jar
./plots/ - Contains the plots generated whilst analysing the results
./xml/ - is a backup of the generate xml files
