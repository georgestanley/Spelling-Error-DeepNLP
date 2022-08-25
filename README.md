# Spelling-Error-Detection

Steps to reproduce:
1. Clone the Github / SVN repo
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

5. GUI

### Input Arguments

| Name            | Datatype | Default Values<br/>Semi-Character with Context | Default Values<br/>Semi-Character without Context        | Default Values<br/>One-hot encoded with context              |
|-----------------|----------|------------------------------------------------|----------------------------------------------------------|--------------------------------------------------------------|
| data_folder     | String   | data                                           | data                                                     | data                                                         |
| output_root     | String   | results                                        | results                                                  | results                                                      |
| input_file      | String   | dev_10.jsonl                                   | top_100_words.json                                       | dev_10.jsonl                                                 |
| val_file        | String   | 'bea60k.repaired.val/bea60_sentences_val_truth_and_false.json| bea60k.repaired.val/bea60_words_val_truth_and_false.json | bea60k.repaired.val/bea60_sentences_val_truth_and_false.json |
| epochs          | int      | 10                                             | 10                                                       | 10                                                           |
| lr              | float    | 0.001                                          | 0.01                                                     | 0.001                                                        |
| bs              | int      | 1000                                           | 1000                                                     | 32                                                           |
| hidden_dim      | int      | 100                                            | 100                                                      | 100                                                          |
| hidden_layers   | int      | 2                                              | 2                                                        | 2                                                            |
| max_len         | int      | _NA_                                           | _NA_                                                     | 60                                                           |
| lower_case_mode | bool     | False                                          | NA                                                       | False                                                        |
| mode            | String   | train                                          | train                                                    | train                                                        |


### Directory Structure

```
|-- Makefile
|-- README.md
|-- application
|   |-- Model.py
|   |-- __init__.py
|   |-- lstm_spell_classifier_w_context.py
|   |-- lstm_spell_classifier_w_context_onehot.py
|   |-- lstm_spell_classifier_wo_context.py
|   |-- run_tests.sh
|   |-- tests
|   |-- utils
|-- bashrc
|-- data
|   |-- bea60k.repaired.test
|   |-- bea60k.repaired.val
|   |-- top_all_words_over_100000_lowercase.json
|   |-- top_all_words_over_200000.json
|   |-- training_1000.jsonl
|   |-- training_200000_lines.jsonl
|   |-- training_5000.jsonl
`-- requirements.txt

```