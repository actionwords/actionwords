# Action Word Prediction for Neural Source Code summarization
This repository contains the code for action word prediction, a tool used for predicting only the action words of a subroutine comment. This project is aimed to demonstrate the importance of action words and how incorrect prediction of action words leads to poorer source code summary prediction overall.

## Dependencies
We assume Ubuntu 18.04, Python 3.6.7, Keras 2.2.4, numpy 1.16.2, Tensorflow-gpu 1.14, javalang 0.12.0, nltk 3.4.

## Action Word Prediction

### Step 1: Obtain Dataset
We use the dataset of 2.1m Java methods and method comments, already cleaned and separated into train/val/test sets by LeClair et al.

(Their raw data was downloaded from: http://leclair.tech/data/funcom/)  

The data for the code2seq and graph2seq models were obtained from Haque et. al (similarly cleaned and separated into train/val/test sets) </br>

(Their raw data was downloaded from: https://github.com/Attn-to-FC/Attn-to-FC)

Extract the dataset to a directory (/nfs/projects/ is the assumed default) so that you have a directory structure:  
/nfs/projects/firstwords/data/standard/dataset.pkl.

The /nfs/projects/firstwords/data also contains an outdir child directory. 
This directory contains the model files, configuration files and prediction outputs of the models.

Therefore, the default directory structure should be: </br>
```/nfs/projects/firstwords/data/standard``` which contains the dataset obtained from LeClair et al. </br>
```/nfs/projects/firstwords/data/graphast_data``` which contains the dataset compatible for code2seq and graph2seq </br>
```/nfs/projects/firstwords/data/outdir``` which contains the model files, configuration files, prediction files and have the following structure:</br>
```
/nfs/projects/firstwords/data/outdir/models/  
/nfs/projects/firstwords/data/outdir/histories/  
/nfs/projects/firstwords/data/outdir/predictions/  
```

For this project, we also created a C/C++ dataset, following the recommendations by Haque et. al. (https://arxiv.org/abs/2004.04881) and LeClair et. al. (https://arxiv.org/abs/1902.01954) which can also be extracted in a similar directory structure as before.

https://ccppdataset1m-nodupes.s3.us-east-2.amazonaws.com/ccppdata.zip

If you choose to have a different directory structure, please make the necessary changes in myutils.py, predict.py and train.py.

### Step 2: Train a Model

```console
you@server:~/dev/firstwords$ time python3 train.py --batch-size=1000 --epochs=10 --model-type=ast-attendgru-fc --data=/nfs/projects/firstwords/data/standard --outdir=/nfs/projects/firstwords/data/outdir --datfile=ccpp1m_dataset.pkl --fwfile=javafirstwords_10.pkl--gpu=0
```

Model types are defined in model.py. All the models implementations are our faithful implementations of the original papers. We thank Haque et. al. for making their implementation public at https://github.com/Attn-to-FC/Attn-to-FC and we made the necessary changes to the model and batch generators to make the models train and predict on the first word of the comment.

```console
you@server:~/dev/firstwords$ time python3 train.py --help
```

This will output the list of input arguments that can be passed via the command line to figure out what information needs to be included to run the train.py file.

### Step 3: Predict Output

```console
you@server:~/dev/firstwords$ time python3 predict.py /nfs/projects/firstwords/data/outdir/models/ast-attendgru-fc_E02_1589046987.h5 --fwfile=javafirstwords_10.pkl --gpu=0 --data=/nfs/projects/firstwords/data/standard --outdir=/nfs/projects/funcom/data/outdir --datfile=ccpp1m_dataset.pkl
```
The above command was used to predict the first words for the test set for the top 10 most commonly occuring action words along with other.

```console
you@server:~/dev/firstwords$ time python3 predict.py /path/to/model/file --help
```

This will output thr list of input arguments that can be passed via the command line to figure out what information needs to be included to run the predict.py file

Note that all these models use CuDNNGRU instead of standard GRU, so a GPU is necessary during both training and prediction for the models.

## Full Sentence Comment Prediction

The paper accompanying this repository also includes experiments to predict full sentence comments. The code for this experiment was run on the models graciously made available online by Haque et. al. at https://github.com/Attn-to-FC/Attn-to-FC. However, we edited their bleu score script to calculate the bleu score based on our project requirements.

## Author information

To learn more about the author, visit https://sakibhaque.github.io/ If you have any questions or concerns, please do not hesitate to reach out the authors via the following email: actionwords.saner2021@gmail.com, shaque@nd.edu

<!--
**actionwords/actionwords** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-->
