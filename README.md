# Spelling-Error-Detection

Steps to reproduce:
1. clone the Github / SVN repo
```
git clone https://github.com/georgestanley/Spelling-Error

svn linkl
```
2. Go to the respective working branch
3. Create a docker image by the following steps:
   1. Build the Image:
   
       ```docker build -t spelling-error-detection .```
   2. Run the container. Here we also have to mount the datasets folder if one is interested in training on the complete dataset.

        ```docker run -it --name spelling-error-detection -v /nfs/students/stanley-george/data:/app/data/ spelling-dummy```


4. Now that code and data is in place, let's try them:
   1. 

### Directory Structure

```
|-- Makefile
|-- application
|   |-- Model.py
|   |-- __init__.py
|   |-- lstm_spell_classifier_w_context.py
|   |-- lstm_spell_classifier_w_context_onehot.py
|   |-- lstm_spell_classifier_wo_context.py
|   |-- requirements.txt
|   |-- run_tests.sh
|   |-- tests
|   |-- utils
|-- bashrc
|-- data



```