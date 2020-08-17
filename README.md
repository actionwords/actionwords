# Action Word Prediction for Neural Source Code summarization
This repository contains the code for action word prediction, a tool used for predicting only the action words of a subroutine comment. This project is aimed to demonstrate the importance of action words and how incorrect prediction of action words leads to poorer source code summary prediction overall.

## Usage

### Step1: Dependencies
We assume Ubuntu 18.04, Python 3.6.7, Keras 2.2.4, numpy 1.16.2, Tensorflow-gpu 1.14, javalang 0.12.0, nltk 3.4.

### Step 2: Obtain Dataset
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

If you choose to have a different directory structure, please make the necessary changes in myutils.py, predict.py and train.py.

### Step 3: Train a Model

```console
you@server:~/dev/firstwords$ time python3 train.py --batch-size=1000 --epochs=10 --model-type=ast-attendgru-fc --data=/nfs/projects/firstwords/data/standard --outdir=/nfs/projects/firstwords/data/outdir --datfile=ccpp1m_dataset.pkl --fwfile=javafirstwords_10.pkl--gpu=0
```

Model types are defined in model.py. All the models implementations are our faithful implementations of the original papers. We thank Haque et. al. for making their implementation public at https://github.com/Attn-to-FC/Attn-to-FC and we made the necessary changes to the model and batch generators to make the models train and predict on the first word of the comment.

### Step 4: Predict Output

```console
you@server:~/dev/firstwords$ time python3 predict.py /nfs/projects/firstwords/data/outdir/models/ast-attendgru-fc_E02_1589046987.h5 --fwfile=javafirstwords_10.pkl --gpu=0 --data=/nfs/projects/firstwords/data/standard --outdir=/nfs/projects/funcom/data/outdir --datfile=ccpp1m_dataset.pkl
```

<!--
**actionwords/actionwords** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
-->
