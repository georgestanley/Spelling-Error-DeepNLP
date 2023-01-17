# Spelling-Error-Detection
**Project Report**

The project report (as a blogpost) can be found at [project-spelling-error-detection.md](project-spelling-error-detection.md)

**Steps to reproduce:**
1. Clone the Github / SVN repo (for people from Chair of Prof. Bast)
```
git clone https://github.com/georgestanley/Spelling-Error

svn co https://ad-svn.informatik.uni-freiburg.de/student-projects/stanley-george

```
2. Go to the respective working branch (Do this only if specifically mentioned by someone)
3. Create a docker image by the following steps:
   1. Build the Image:
   
       ```docker build -t stanley-george-project .```
   2. Run the container. Here we also have to mount the datasets folder if one is interested in training on the complete dataset.

        ```docker run -it --gpus=all --name stanley-george-project -v /nfs/students/stanley-george/data:/app/data/ stanley-george-project```
   
   **Note**: To remove an existing image with the same name, use the command:
        
   ```docker rm stanley-george-project```
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

5. Makefile
   1. Type `make` to see the possible options
   2. Type `make start_app_console` to start the app in the console mode
   3. Type `make start_app_webapp` to start the app as a Webapp made using Dash (Go to localhost:8050)
   4. Type `make start_app_file_eval` to start the app and test a text file

### Input Arguments

| Name            | Datatype | Default Values<br/>Semi-Character with Context                | Default Values<br/>Semi-Character without Context           | Default Values<br/>One-hot encoded with context              |
|-----------------|----------|---------------------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------|
| data_folder     | String   | data                                                          | data                                                        | data                                                         |
| output_root     | String   | results                                                       | results                                                     | results                                                      |
| input_file      | String   | dev_10.jsonl                                                  | top_100_words.json                                          | dev_10.jsonl                                                 |
| val_file        | String   | 'bea60k.repaired.val/bea60_sentences_val_truth_and_false.json | bea60k.repaired.val/bea60_words_val_truth_and_false.json    | bea60k.repaired.val/bea60_sentences_val_truth_and_false.json |
| epochs          | int      | 10                                                            | 10                                                          | 10                                                           |
| lr              | float    | 0.001                                                         | 0.01                                                        | 0.001                                                        |
| bs              | int      | 1000                                                          | 1000                                                        | 32                                                           |
| hidden_dim      | int      | 100                                                           | 100                                                         | 100                                                          |
| hidden_layers   | int      | 2                                                             | 2                                                           | 2                                                            |
| max_len         | int      | _NA_                                                          | _NA_                                                        | 60                                                           |
| lower_case_mode | bool     | False                                                         | NA                                                          | False                                                        |
| mode            | String   | train                                                         | train                                                       | train                                                        |
| eval_model_path | String   | trained_models/semi_character_w_context.pth                   | trained_models/semi_character_wo_context.pth                | trained_models/onehot_w_context.pth                          |   
| eval_file       | String   | bea60k.repaired.test/bea60_sentences_test_truth_and_false.json| bea60k.repaired.test//bea60_words_test_truth_and_false.json | bea60k.repaired.test//bea60_sentences_test_truth_and_false.json|


### Directory Structure

```
|-- Makefile
|-- README.md
|-- Dockerfile
|-- results
|-- runs
|-- application
|   |-- Model.py
|   |-- __init__.py
|   |-- app.py
|   |-- lstm_spell_classifier_w_context.py
|   |-- lstm_spell_classifier_w_context_onehot.py
|   |-- lstm_spell_classifier_wo_context.py
|   |-- run_tests.sh
|   |-- tests
|   |-- utils
|-- bashrc
|-- data
|-- trained_models
`-- requirements.txt

```
Folder/file definitions


| Folder         | Description                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------|
| results        | The output folder for an experiment gets generated here and houses the saved models and log files |
| runs           | Contains the tensorboard metrics logs.                                                            |
| application    | Contains the important codes related to our application                                           |
| data           | The data files needed which serve as input to our models                                          |
| trained_models | Contains the pretrained models which can be used for evaluation purpose.                          |






