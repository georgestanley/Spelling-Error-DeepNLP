FROM anibali/pytorch:1.11.0-cuda11.5-ubuntu20.04
LABEL maintainer="Stanley George <george@tf.uni-freiburg.de>"
#RUN apt-get update && apt-get install -y make vim && rm -rf /var/lib/apt/lists/*
USER root
RUN apt-get update
RUN apt-get install -y make vim
RUN rm -rf /var/lib/apt/lists/*

COPY Makefile Makefile
COPY bashrc bashrc
COPY requirements.txt requirements.txt
COPY application//__init__.py application//__init__.py
COPY application//lstm_spell_classifier_wo_context.py application//lstm_spell_classifier_wo_context.py
COPY application//lstm_spell_classifier_w_context.py  application//lstm_spell_classifier_w_context.py
COPY application//lstm_spell_classifier_w_context_onehot.py application//lstm_spell_classifier_w_context_onehot.py
COPY application//app.py application//app.py
COPY application//Model.py application//Model.py
COPY application//utils application//utils
COPY application//tests application//tests
COPY application//run_tests.sh application//run_tests.sh
COPY trained_models trained_models
COPY README.md .
RUN chmod +x application//run_tests.sh
EXPOSE 8050

#COPY data/bea60k.repaired/ data/bea60k.repaired/
#COPY data/bea60k.repaired.test/ data/bea60k.repaired.test/
#COPY data/bea60k.repaired.val/ data/bea60k.repaired.val/
#COPY data/top_all_words_over_200000.json data/top_all_words_over_200000.json
#COPY data/dev_10.jsonl data/dev_10.jsonl
RUN pip install -r requirements.txt

CMD ["/bin/bash", "--rcfile", "bashrc"]

# docker build -t stanley-george-spelling-error-detection .
# docker run -it -v $(pwd)/path/to/input:/input:ro -v $(pwd)/path/to/output:/output:rw --name stanley-george-spelling-error-detection stanley-george-spelling-error-detection
# docker run -it gpus all stanley-george-spelling-error-detection .