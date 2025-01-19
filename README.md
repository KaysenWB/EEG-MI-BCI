

# Introduction
This motor imagery brain-computer interface and EEG decoding process uses only convolutional networks. The data used is the 2a dataset of BCI Competition IV, which contains four motor imagery classes: left hand, right hand, foot, and tongue. The raw data can be downloaded from the official website of the BCI competition. The preprocessed data used in this work can be downloaded from: https://github.com/bregydoc/bcidatasetIV2a.git. This work uses 22 electrode data from 9 subjects, with 2264 valid trials, including 1827 training trials (consisting of 80% of each subject's trials) and 437 testing trials, with a training and testing data ratio of 8:2.

This work evaluates the decoding performance using accuracy, Kappa, and F1-score. In the test, the trained model achieves an average of 0.6906, 0.5787, and 0.6785 for the three indicators. We provide the test outputs and the trained model in the ‘results’ folder.

# result
![Figure01](https://github.com/KaysenWB/EEG-MI-BCI/blob/main/EEG-BCI/results/figure/metrics.png?raw=true)
![Figure02](https://github.com/KaysenWB/EEG-MI-BCI/blob/main/EEG-BCI/results/figure/trials.png?raw=true)
![Figure03](https://github.com/KaysenWB/EEG-MI-BCI/blob/main/EEG-BCI/results/figure/trials_correct.png?raw=true)

