# Hebrew syllabification
Description: Final project in "Intro. to NLP" course.

## Schedule
<!--- :heavy_check_mark: --->
| Due date | Task | Status | Date | 
| --- | --- | ---| ---|
| 22/7 | Finish script to convert hebrew to transl. chars | :heavy_check_mark: | 19/7 |
| 26/7 | Annotate some data for HMM work |:heavy_check_mark:| 24/7 |
| 2/8 | Unigram HMM code finished |:heavy_check_mark:| 24/7 |
| 2/8 | Bigram, Trigram HMM code finished |:heavy_check_mark:| 25/7 |
| 2/8 | Metrics for HMM finished |:heavy_check_mark:| 25/7 |
| 2/8 | Division to syllables finished |:heavy_check_mark:|7/8|
| 9/8 | Conversion to english letters finished |:heavy_check_mark:|7/8|
| 15/8 | CRF code finished |:heavy_check_mark:| 28/7 |
| 8/9 | Final data version - no more annotation from this point |:heavy_check_mark:||
| 9/8 | Implement edit distance metric |:heavy_check_mark:|7/8|
| 16/8 | Finish NN |:heavy_check_mark:|15/8|
| 19/9 | Finished project |:heavy_check_mark:||
| 21/9 | Verified project |||

# Files description
## Models
 - crf_sentence.py - CRF model for word and sentence features
 - crf_word.py - CRF model for word only features
 - embedding_mds.py - Create embedding matrix using MDS
 - embedding_nn.py - Create embedding matrix using NN
 - hmm.py - HMM model
 - memm.py - MEMM model
 - rnn.py - RNN model
 
## Utilities
 - metrics.py - Data metrics

test.py
input_proc\utils.py
input_proc\verifier.py
post_proc\post_processing.py
post_proc\statistics.py
post_proc\syllabification.py
post_proc\test_scores.py
post_proc\utils.py
