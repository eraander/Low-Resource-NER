-Basic Structure-
The main code is in n_sami_entities.py.

The code in that file follows a similar format to the homeworks, since we based the basic approach on that. We used the same annotation format, so the method for ingesting the .json doc with the annotations for the articles remains the same. 

In addition to some of the feature classes, we’ve added some features. One extracts suffixes (the last 2 characters when a token is longer than 5 characters). There is also a name list feature that adds “name[index]” : 1.0 to the feature vector if the token is a name on the name list. We use this with nameless for Sami and Finnish given names. Additionally, since we used Northern Sami word vectors we added a word vector feature that is similar in structure to the one we used for english, but just tweaked a little for the specifics of the format for the N. Sami vectors. So, there are two word vector features included (WordVectorFeature which is used for N. Sami and WordVectorFeatureEng which is used for English).

There are a few additional functions right above the main() functions that are small things to help in the main code. getVectors is used to get the vector information from the text files, addPRF and divPRF are used to add or divide the precision, recall, and F1 for two PRF1 objects. This was useful because in calculating the accuracy, we were using cross-validation and it was simpler to add and divide that way to get the overall scores.

-Main Methods-

The two main methods are where the bulk of the changes are made. main_sami() runs the model on the N. Sami data, and main_engl() runs the model on the same amount of English data for comparison. If the code is run now as is, the sami model will run with the features that gave us the best results. To see the results for English, that can be un-commented at the very end.

~main_sami~
The first chunk of code reads in the .jsonl file with the data we compiled and annotated and creates SpacyDocs with the spans for the entities based on the annotations. This is followed by the code for the model and the features. The features we didn’t use in our final implementation are commented out, but to use them they can be uncommented and the code should run as long as the files required for the embeddings and name lists are in the same directory.The code following that (after the docs are shuffled) breaks the data up into 5 folds of 10 for cross-validation, then runs the folds through the program, training 4 and testing one each time. The scores from each iteration are collected and averaged. Then below that were the print statements start is the code that is printing our results. 

~main_engl~
This follows the same structure as main_sami, except that the code reading in the file is different since we were provided with a way to read in the conll data that we used for English data. (load_conll2003() will read in the English data and create the Spacy docs). There is also a difference in some of the features based on what goes with the English data (no name lists and a different word vector feature using the word vectors from ‘wiki-news-300d-1M-subword.magnitude’). But again, all of the features that we didn’t use for our final results are commented out.

-Additional files-

hw3utils.py & the cosi217 folder- these are the same files (unaltered) that we used for the homeworks and they works in conjunction with n_sami_entities.py.

cupy_utils.py & embeddings.py- this is open source code from the same location as the word vectors. It is used in n_sami_entities.py to read in the word vectors.

