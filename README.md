# README
The replication package of paper “Are your comments outdated? Towards automatically detecting code-comment consistency.”

## Download resource package 
Google Cloud Disk Link, https://drive.google.com/drive/folders/12xYfd8DC66OdBy3HhZs2T8qrGBfVEPCo?usp=sharing

**Resource Description**

1.features, it contains the processed CCSet datasets of method and block types. You can view the datasets we use here.

2.word2vector, including word vector training corpus and trained word vector model.

3.OCD.zip, trained OCD model.

4.ocd_data.zip, the dataset used for the baseline OCD is the same as the dataset used in our project, just the input form is different.

5.RQ5, the dataset used in RQ4 includes ejbca, freecol and opennms, among which detected_ Outdated.txt is the list of obsolete comment files detected, which can be found in the *feature folder respectively.  *git folder saves the original git clone file of the project. Questionnaire and results are in this folder as well.

6.python_raw_ccset, manually collected python dataset for RQ4.

## Train and test the model
Run */outdate_predict/main.py*

This includes grid search, which takes a long time.
## Classifier comparison and calibration
Run */classifiers/main.py* （RQ2)

## Utils
1.convert_co_CCSet, tools used to generate CCSet from original commit.

2.get_change, tools for extracting changes from CCSet.

3.extract_features, tools for feature extraction.

4.generate_table, tools for generating csv files.

5.data_analyse_tool, tools for analyzing datasets.

6.drop_duplicates_tool, tools used to remove duplicates of original data.

7.word2vector, tools for training word vectors.
## Baseline
**Rule based baseline,**
put the folder "features" in resource package in the CoCC code directory, and then run *baseline/rule/rule.py*

**OCD,**
open the *CoCC/baseline/OCD* folder, and unzip OCD.zip and ocd_data.zip to the current directory, and then run the follow in the command line.
```python
python -m infer --log-dir OCD --config configs/OCD.yml

python -m eval --log-dir OCD --config configs/OCD.yml
```
