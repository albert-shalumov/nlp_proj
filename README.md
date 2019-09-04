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
| 8/9 | Final data version - no more annotation from this point |||
| 9/8 | Implement edit distance metric |:heavy_check_mark:|7/8|
| 16/8 | Finish NN |:heavy_check_mark:|15/8|
| 19/9 | Finished project |||
| 21/9 | Verified project |||

## File description
 - hmm.py - Contains HMM implementation (adapter for NLTK). If called directly, scans over different configurations:
 ngram: [1..9], smoothing:{MLE, Laplace, Add-\delta [0.1,0.2,...0.9], GoodTuring}. Seed is fixed to 0 to ensure 
 reproducability.
 - memm.py - Contains MEMM implementation (using sklearn for optimization). If called directly,
 scans over combinations of possible features: ['IS_FIRST', 'IS_LAST', 'IDX', 'VAL', 'PRV_VAL', 'NXT_VAL', 'FRST_VAL', 'LST_VAL', 'SCND_VAL', 'SCND_LST_VAL', 'LEN'].
 Seed is fixed to 0 to ensure reproducability. 
 - crf_word.py - Contains CRF implementation using word scope only. If called directly,
 scans over combinations of possible features: ['IS_FIRST', 'IS_LAST', 'IDX', 'VAL', 'PRV_VAL', 'NXT_VAL', 'FRST_VAL', 'LST_VAL', 'SCND_VAL', 'SCND_LST_VAL', 'LEN'].
 Seed is fixed to 0 to ensure reproducability.  
