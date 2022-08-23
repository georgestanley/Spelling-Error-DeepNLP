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
   
   **Note**: To remove an existing image with the same name, use the command:
        
   ```docker rm spelling-error-detection```
4. Now that code and data is in place, let's try them:
   1. Train a spell-classifier (from `application` folder).

      For E.g. To train the LSTM spell classifier with context and Semi-Character encoding, execute the command:
      
      `python -m application.lstm_spell_classifier_w_context --data_folder=data`

      A table of the different input parameters that can be provided are listed in Section xx

   2. Evaluate of the test dataset.

      Execute the same file as in step (i) except set the mode to `test`

      `python -m application.lstm_spell_classifier_w_context --mode=test`
   3. Run all the Unit-tests.
      
      Execute the shell script which triggers all unit-test files as: `./application/run_tests.sh`
   4. Run individual unit-tests.

      All the uni-tests are in the application/tests folder. You can trigger them as below from the parent directory:
   
      `python -m application.tests.test_lstm_spell_classifier_w_context`

6. GUI

### Input Arguments

| Name        | Datatype | Default Values<br/>Semi-Character with Context | Default Values<br/>Semi-Character without Context | Default Values<br/>One-hot encoded with context |
|-------------|----------|------------------------------------------------|---------------------------------------------------|-------------------------------------------------|
| data_folder | String   | data                                           |                                                   |                                                 |
| output_root | String   |                                                |                                                   |                                                 |
| input_file  | String   |                                                |                                                   |                                                 |
| val_file    | String   |                                                |                                                   |                                                 |
| epochs      | int      | 10                                             |                                                   |                                                 |
| lr          | float    | 0.001                                          |                                                   |                                                 |
| bs          |          |                                                |                                                   |                                                 |



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