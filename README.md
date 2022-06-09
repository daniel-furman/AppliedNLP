# anlp21

Course materials for "Applied Natural Language Processing" (INFO 256, Fall 2021, UC Berkeley)
All Lectures + Syllabus: http://people.ischool.berkeley.edu/~dbamman/info256.html

## Relavent terms

|Term|Definition|
|---|---|
|Corpus|collection of texts|
|Tokenization|breaks unstructured data and natural language text into chunks of information that can be considered as discrete elements|
|Lemmatization|map tokens to their base word or a root word, which forms the basis for other words. For example, the lemma of the words “playing” and “played” is “play”|
|Stemming|map tokens to an equal or smaller form ot the word|
|Part of Speech Tagging|labeling of the words in a text according to their word types (noun, adjective, adverb, verb, etc.)|
|Type-token ratio|measure of text complexity between two equal lengthed documents|
|Log odds ratio|common method for finding distinctive terms in two datasets, optionality for including priors of word frequency from a reference corpus|
|Word embedding|a real number, vector representation of a word|
|Sequence embedding|a real number, vector representation of a sequence / sentence|
|Word senses|one of the meanings of a word|
|WordNet|a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept|
|Coreference|two or more expressions refer to the same person or thing, useful for timeline generation for example|

## Notebook documentation

|Notebook|Description|
|---|---|
|1.words/EvaluateTokenizationForSentiment|The impact of tokenization choices on sentiment classification.|
|1.words/exploreTokenization|Different methods for tokenizing texts (whitespace, NLTK, spacy, regex)|
|1.words/TokenizePrintedBooks|Design a better tokenizer for printed books|
|1.words/Text_Complexity|Implement type-token ratio and Flesch-Kincaid Grade Level scores for text|
|2.compare/ChiSquare, Mann-Whitney Tests|explore two tests for finding distinctive terms|
|2.compare/Log-odds ratio with priors|Implement the log-odds ratio with an informative (and uninformative) Dirichlet prior|
|3.dictionaries/DictionaryTimeSeries|Plot sentiment over time using human-defined dictionaries|
|3.dictionaries/Empath|explore using Empath dictionaries to characterize texts|
|4.embeddings/DistributionalSimilarity|explore distributional hypothesis to build high-dimensional, sparse representations for words|
|4.embeddings/WordEmbeddings|explore word embeddings using Gensim|
|4.embeddings/Semaxis|Implement SemAxis for scoring terms along a user-defined axis (e.g., positive-negative, concrete-abstract, hot-cold),
|4.embeddings/BERT|explore the basics of token representations in BERT and use it to find token nearest neighbors|
|4.embedings/SequenceEmbeddings|Use sequence embeddings to find TV episode summaries most similar to a short description|
|5.eda/WordSenseClustering|Inferring distinct word senses using KMeans clustering over BERT representations|
|5.eda/Haiku KMeans|explore text representation in clustering by trying to group haiku and non-haiku poems into two distinct clusters|
|6.classification/Classification|explores feature engineering for text classification, training a regularized logistic regression model with scikit-learn for the binary classification task of predicting a movie's genre.|
|6.classification/FeatureExploration|explores feature engineering for text classification with dictionary features and unigram features.|
|6.classification/Hyperparameters|explores text classification, introducing a majority class baseline and analyzing the effect of hyperparameter choices on accuracy.|
|7.regression/Regularization|This notebook explores linear regression with L2 (ridge) and L1 (lasso) regularization.|
|8.tests/Bootstrap|explores the use of the bootstrap to create confidence intervals for any statistic of interest that is estimated from data.|
|8.tests/ParametricTest|explores a simple hypothesis test checking whether the accuracy of a trained model for binary classificadtion is meaningfully different from a majority class baseline. We test this making a parametric assumption: we assume that the binary correct/incorrect results follow a binomial distribution (and approximate the binomial with a normal distribution).|
|8.tests/PermutationTest|This notebook explores the use of the permutation test to assess the significance of coefficents learned in logistic regression (testing against the null that each  𝛽  = 0).|
|9.neural/BertClassification|explores using BERT for text classification.|
|9.neural/CNN|explores using CNN for binary text classification using the pytorch library.|
|9.neural/FFNN|explores logistic regression and feed-forward neural networks for binary text classification for your text classification problem, using the pytorch library.|
|9.neural/Interpretability|explores integrated gradients to identify the tokens in the input that are most responsible for the predictions that a bigram CNN model is making. Before running, install the captum library.|
|9.neural/PromptDesign|explores few-shot learning with GPT-2. While GPT-2 is a less expressive model than GPT-3 (and hence not as a good of a few shot learner), it can fit within the memory and processing constraints of laptops while also being openly available. Can you create a new classification task and design prompts to differentiate between the classes within it?|
|10.wordnet/ExploreWordNet|explores WordNet synsets, presenting a simple method for finding in a text all mentions of all hyponyms of a given node in the WordNet hierarchy (e.g., finding all vehicles in a text).|
|10.wordnet/Lesk|explores the lesk algorithm, which infers the word sense of a given word in the sentence.|
|11.pos/POS_tagging|explores part of speech tagging: categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context.|
|12.ner/ToponymResolution|explores named entity recognition through the lens of toponym resolution, using NER to extract a list of geopolitical place names in a text, and then plotting those locations on a map.|
|12.ner/ExtractingSocialNetworks|explores the task of extracting social networks from text: for a given set of people mentioned in a text, can we extract a social network that connects them? In this social network, people are the nodes, and the edges between them are weighted by the strength of their connection. How you define what "connection" means here is up to you (within reason).|
|13.mwe/JustesonKatz95|Explores identifying multiword expressions using the part-of-speech filtering technique of Justeson and Katz (1995), "Technical terminology: some linguistic properties and an algorithm for identification in text". Functionality to replace multiword expressions with a single token (poor man -> poor_man)|
|14.syntax/SyntacticRelations|explores dependency parsing by identifying the actions and objects that are characteristically associated with male and female characters.|
|15.coref/CorefSetup|this notebook explores the neuralcoref spacy package. *Only compatible with spacy 2*|
|15.coref/ExtractTimeline|explores coreference resolution for the task of timeline generation: for a given biography on Wikipedia, can you extract all of the events associated with the people mentioned and create one timeline for each person.|
|16.ie/DependencyParsing|explores relation extraction by measuring common dependency paths between two entities that hold a given relation to each other -- here, the relation "born_in" between a PER entity and an GPE entity, using data from Wikipedia biographies.|
|17.sequence_alignment/Smith-Waterman-Alignment|explores the Smith-Waterman algorithm to find local regions of alignment between two pieces of text.|




