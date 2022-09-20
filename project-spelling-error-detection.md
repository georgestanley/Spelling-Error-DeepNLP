---
title: "Spelling Error Detection"
date: 2022-08-01T15:36:35+05:30
author: "Stanley George"
authorAvatar: "img/ada.jpg"
tags: ["deep learning","spelling error","LSTM"]
categories: []
image: "img/writing.jpg"
draft: true
---
This project aims to detect spelling errors in a given sentence using a Deep Learning approach.

<div>
<p><b>CONTENTS</b></p>
<ol>
<li><a href="#div1">Introduction</a>
<li><a href="#div2">Problem Statement</a>
<li><a href="#div3">Dataset</a>
<li><a href="#div4">Data Encoding</a>
<li><a href="#div5">Models</a>
<li><a href="#div6">Dataset preparation</a>
<li><a href="#div7">Training</a>
<li><a href="#div8">Results</a>
<li><a href="#div9">Observations</a>
<li><a href="#div10">Future Works</a>
</ol>
</div>
<ol>
<p>
<div id ="div1">
<b><li>INTRODUCTION </b><br>
The advent of computers into almost all spheres of human-life has made us more dependent on the keypad to convey information rather than filling out forms by hand. 
The use of a keypad (physical or virtual as in a smartphone) has enabled us to read legibly unlike the instance when one can't decipher the handwriting inscribed on a letter. 
But, human beings are still prone to make spelling errors. In this project, we try to detect spelling errors in an English language input text.

</div>
<br>
<div id="div2">
<li><b>PROBLEM STATEMENT:</b></li> 
Given a Sentence in English language as input, output the words that have been incorrectly spelled.

</div>

<br>
<div id="div3">
<b><li>DATASETS</li> </b>
In this project, we used two datasets for training and evaluation respectively. We decided to use two datasets so as to ensure our model has been able to generalise better instead of simply memorizing all words.
<ol type="i">
<li><b>Wikipedia Dataset</b></li>
In this project, the Wikipedia dataset (collected by the AD Chair) was used for training the models. The datasets were already split up as development, training and testing files - however we used only the training file for our model's training task.
<p><b>Motivation to choose it :</b>Since Wikipedia is a public website, we expect less spelling mistakes. Moreover, it also offers articles from a wide variety of topics.
This is in contrast to providing as input a JSON file containing an open-sourced book. 
Even though the contents would be well-proofed from grammatical and spelling errors, it however will be limited to small context of words on which the book has been based.

<p>As part of the initial investigations, some other datasets too were considered which can also serve the same purpose.
Some of them that can be used by future students who are conducting research on similar areas are:
<ul>
<li> <a href="https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection">Reuters News Dataset</a> </li> 
<li> <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41880.pdf" >Google's 1 Billion Word dataset</a>  </li>
</ul>

<li><b>BEA-60k</b></li>
It is a dataset collected by the team who developed the <a href="https://github.com/neuspell#Installation" > NeuSpell </a> spelling correction toolkit. It is made up of close to 70K spelling errors in around 63000 sentences.
<p> For this project, we splitted the data equally (50-50) as our validation and test set.
</ol>
</div>

<div id="div4"><li><b>Data Encoding</b></li>
An important task when working with textual data for a Deep Learning model (or any Machine Learning model) is to decide on the encoding technique. We need a mechanism by which we convey our textual data into a numerical format which our models can read and learn to optimise.
As part of this project we experimented with two such encoding techniques as discussed below:
<ol>
<li><b>Semi-Character Vector</b></li>
This technique was introduced in the 2017 paper by Sakaguchi et. al in their paper titled <a href="https://arxiv.org/pdf/1608.02214.pdf">Robsut Wrod Reocginiton via Semi-Character Recurrent Neural Network</a>.
Human beings often can understand words in a sentence even if a word is misspelled.
Here the authors used on the research conducted by other Scientists to understand the level of difficulty faced by humans to decipher a word based on the position at which a spelling error occurred in a word.

<p>Consider the below three senetences.
<ol>
<li><i>The boy cuold not slove the probelm so he aksed for help.</i></li>  
<li><i>The boy coudl not solev the problme so he askde for help.</i></li>
<li><i>The boy oculd not oslve the rpoblem so he saked for help.</i></li>
</ol>

It was found that reading sentence (a) and (b) were comparetively easier than (c) because it was the BEGinning characters that were jumbled. 
In case of (a), the INTernal words were swapped whereas in (b) the ENDing characters were altered. 
Overall, the difficulty in reading jumbled words can be summarized as: N ≤ INT < END < BEG where N denotes No error. 
Thus we see that the beginning letter and the ending letter have more importance in human word recognition. 

<p>We use this principle to construct a word vector which is made up of three sub-vectors (bn, in, en) that correspond to the characters position. The first and third sub-vectors represent the first and last character of the n-th word. They are same as the one-hot representations.
The second sub-vector _in_ represents the character count of each word except the first and last word.
Refer the below figure for a sample word vector for the word 'Dictionary'.
<img src="assets/semi_character_example.png">
<p>So, considering we have a five word sentence (e.g. My favourite dictionary is Oxford) as input to the model with a Vocabulary set of 52 elements  (English alphabets in lower and upper case), 
we have a word vector of shape 5* 156 (5 words * (bn+in+en)) which is passed as input to our model. 

<li><b>Character level One-hot Encoding</b></li>
<p>One-hot encoding is often used in the field of Machine learning to encode categorical data into numerical data.
It is a binary representation vector wherein a categorical class' index is assigned a positive value (1) and rest all indexes have a negative (0) value.
<p>However, we can extend this concept further and use it to encode a Word at Character level where our Vocabulary set is made up of the alphabets of the language. This is same as the bn and en vectors created in section <a href="#div4">4.i</a>

<p>So, considering the same five word sentence used for the Semi-Character example (i.e. My favourite dictionary is Oxford) , 
we will have a One-hot vector of shape 33*53 where 33 is length of the sentence and 53 is the length of the vocabulary.
This means that every single character has a vector associated with it. 
In the vocabulary set we add an additional vector to account for Spaces in the sentence. Hence, the vocabulary length now becomes 52+1 = 53.

</ol>
</div>

<div id="div5"><li><b>Models:</b></li>
As part of this project, we mainly used the LSTM networks for our analysis. Below we give a briefly introduce what are Recurrent Neural Networks (RNN) and LSTM networks.

<ol>
<li><b>Recurrent Neural Networks:</b></li>
Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:
<figure>
<img src="assets/rnn_demo_stanford.png">
<figcaption>Fig 1: An example of the Semi-Character Vector for the word Dictionary</figcaption>
</figure>
<p>Such networks are mainly used in the field of Natural language processing and Speech recognition. 
However, RNN networks also experience a phenomena known as exploding/vanishing gradient problem. It happens when the network finds it difficult to capture long term dependencies because of multiplicative gradient that can be exponentially decreasing/increasing with respect to the number of layers.

<li><b>Long Short-term Memory:</b></li>

LSTMs solve the problem of vanishing gradient by introducing gates inside the network that regulate the flow of information within a cell.
Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.
Due to this, networks can now learn from sequences that are of longer length. 
<p>If you would like to learn more about LSTM and how they function, check out this well explained blog by Chris Olah 
<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTM Networks</a> where he explains how each LSTM modules operate with good visual cues.


</ol>

<p><i>Note: The project began with experimenting on Multi-Layer Perceptrons (MLP) and Recurrent Neural Networks(RNN). However, since the results in the initial stages were not very promising, we dedicated more focus towards LSTM networks. Hence, only LSTM evaluation results are listed below.</i>
</div>

<div id="div6">
<li><b>Dataset Preparation:</b></li>
<ol>
<p>
<li><b>Training:</b></li>
Our experiments were focused on two angles: <b>context</b> based and <b>non-context</b> based. 
As the name suggests, in the non-context based approach, the model is trained only on individual words. 
The input was not a sentence of words but just a single word.
For the non-context based approach, the model was trained on the target word and the contextual words that appeared before and after it.
For our experiments, the models were shown two context words before and after the target word.  
<p>Example: In the sentence <i style="color:green">We need <b style="color:red">suffcient</b> carbohydrates in our body </i> , for the target word 
<b style="color:red">sufficient</b>, the input sentence would be  <i style="color:green">We need <b style="color:red">suffcient</b> carbohydrates in </i> i.e. the two sequential words before and after the target.

<p>For the non-context based approach, we extracted all the individual words from 200,000 Wikipedia articles and filtered out the words which occurred less than 20 times. This additional filtering had to be done to remove words that were misspelt or words that occured very rarely in the encyclopedia. The final training dataset contained 1,47,011 words.
<p>As for the context based approach,we didn't need to do any extra preparation tasks. However, due to the limitations with GPU compute, we trained only on a randomly selected corpus of 5000 wikipedia articles. The final dataset contained 17,43,076 5-gram pairs.

<p>For the one-hot encoding technique, we also needed to decide on the length of the one-hot encoded vector. For the same, we plotted a distribution of the length of 5-word sentences of the entire dataset (Fig 1). Based on the results, we decided to set 60 characters as the maximum length of the vector. So, any 5-word sentences greater than 60 characters would be trimmed to 60 characters and sentences padded shorter than 60 would be given extra right-end paddings.
<img src="assets/img_1.png">

<li>Real-time Error generation:</li>
<p>Since we are following a Supervised learning approach, our models need to be trained to distinguish between Positive and Negative Samples by having training it on Positive and Negative Samples.
However, we assume that our dataset is a clean dataset i.e. without any errors. Hence, we introduce errors manually during the training epochs.
</p>

<p>We introduce randomly either of the below three error on the target word:
<ul>
<li>Replace a character with another random character</li>
<li>Introduce an extra character at a random position </li>
<li>Remove a character from a random position</li>
</ul>

Because of this, in every epoch, the network sees a different negative word thereby avoiding a possible overfitting.

<li>Evaluation:</li>
<p>
The BEA-60k dataset was modified as collection of positive and negative sample of 5-word sentences. 
So,the final dataset size was


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-46ru{background-color:#96fffb;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-elvq{background-color:#fffc9e;border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-fymr">Samples With Error</th>
    <th class="tg-fymr">Samples Without Error</th>
    <th class="tg-fymr">Total Samples</th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fymr" rowspan="2">Context-Based (i.e. Sentences)</td>
    <td class="tg-46ru">10,865</td>
    <td class="tg-46ru">10,781</td>
    <td class="tg-46ru">21,646</td>
    <td class="tg-46ru">Validation Set</td>
  </tr>
  <tr>
    <td class="tg-elvq">11,220</td>
    <td class="tg-elvq">11,202</td>
    <td class="tg-elvq">22,422</td>
    <td class="tg-elvq">Test Set</td>
  </tr>
  <tr>
    <td class="tg-fymr" rowspan="2">Non-Context (i.e. Words)</td>
    <td class="tg-46ru">13,899</td>
    <td class="tg-46ru">6,384</td>
    <td class="tg-46ru">20,283</td>
    <td class="tg-46ru">Validation Set</td>
  </tr>
  <tr>
    <td class="tg-elvq">14,299</td>
    <td class="tg-elvq">6,671</td>
    <td class="tg-elvq">20,970</td>
    <td class="tg-elvq">Test Set</td>
  </tr>
</tbody>
</table>
</ol>

</div>

<div id = "div7"><li><b>Training:</b></li>
The models were trained on 4 Nvidia Titan X (Pascal) GPUs. The below table lists some of the important hyperparameters used during the training.


<table>
<tr>
<th></th>
<th>LSTM without context</th>
<th>LSTM with context</th>
<th>LSTM with context</th>
</tr>

<tr>
<td>encoding</td>
<td>Semi-Character</td>
<td>Semi-Character</td>
<td>One-hot encoding</td>
</tr>

<tr>
<td>Epochs</td>
<td></td>
<td></td>
<td></td>
</tr>

<tr>
<td>Learning Rate</td>
<td>0.01</td>
<td>0.001</td>
<td>0.001</td>
</tr>

<tr>
<td>Optimizer</td>
<td>ADAM</td>
<td>ADAM</td>
<td>ADAM</td>
</tr>

<tr>
<td>Loss</td>
<td>Cross Entropy Loss</td>
<td>Cross Entropy Loss</td>
<td>Cross Entropy Loss</td>
</tr>

<tr>
<td>Training time(per Epoch)</td>
<td>1 min</td>
<td>16 min</td>
<td>20 min</td>
</tr>

<tr>
<td>Dataset Size</td>
<td>~150k words </td>
<td>5000 Wikipedia articles</td>
<td>5000 Wikipedia articles</td>
</tr>

</table>

<img src="assets/pic_all_graphs.png">

</div>

<div id="div8"><li><b>Results:</b></li>
The table below shows the accuracy that was achieved on the test dataset and the corresponding F1 Score. Further we also have plotted the Confusion Matrix for the same dataset. lIt is clearly evident that Context based models outperfom non-context based model.  
<table>

<tr>
<th></th>
<th>Accuracy</th>
<th>F1 Score</th>
</tr>

<tr>
<td>LSTM without Context</td>
<td>76.01%</td>
<td>0.798</td>
</tr>

<tr>
<td>LSTM with Context (Semi-Character)</td>
<td>87.90%</td>
<td>0.879</td>
</tr>

<tr>
<td>LSTM with Context (One-hot)</td>
<td>87.77%</td>
<td>0.874</td>
</tr>
</table>


<img src="assets/cm_lstm_wo_context.png" style="align-content: center">
    <figcaption>Confusion Matrix for LSTM Without Context</figcaption>


<figure>
<img src="assets/cm_lstm_w_context_ckpt43.png">
    <figcaption>Confusion Matrix for LSTM With Context Semi Character</figcaption>
</figure>

<figure>
<img src="assets/cm_lstm_onehot_ckpt37.png">
    <figcaption>Confusion Matrix for LSTM With One-Hot Encoding</figcaption>
</figure>



<p><b>Examples of Good and Bad evaluations:</b>

<table>
<tr>
<td>Test Sentence:</td> 
<td>The quick brown fox jumps Over the lazy dog</td>
</tr>

<tr>
<td>LSTM with context (Semi-Character)</td> 
<td>fox</td>
</tr>

<tr>
<td>LSTM with context (One-hot)</td> 
<td>The,quick,fox</td>
</tr>

<tr>
<td>LSTM without context<</td> 
<td>over</td>
</tr>

</table>

Another Example:

<table>
<tr>
<td>Test Sentence:</td> 
<td>An impartial invstigation into the crash was conducted by the agency</td>
</tr>

<tr>
<td>LSTM with context (Semi-Character)</td> 
<td>invstigation</td>
</tr>

<tr>
<td>LSTM with context (One-hot)</td> 
<td>An,impartial,invstigation</td>
</tr>

<tr>
<td>LSTM without context<</td> 
<td>invstigation,was</td>
</tr>

</table>

</div>

<div id="div9">
<li>Observations:</li>
<ul>
<li><b>Does Context really Matter ?</b></li>As seen from the results in Section 7 and on playing with the console (or webapp), one can easily see that having context improves the score. However, let us compare some examples from the test set.
TODO: Find examples from test set where Context was +ve but w/o context was -ve
<li><b>One-hot encoded vectors are faster</b></li>
A closer look at the training plots placed in <a href="#div8">Section 8</a> shows the orange line reaching better metric levels faster than the semi-character encodings. A possible reason for lies in the underlying concept of one-hot vectors. ????
<li><b>The Case matters</b></li>At the beginning of writing my code, I was under the assumption that the models would be happy with just the spellings. But just a trial of removing the .lower() function resulted in a huge spike in the validation accuracy (~10%).  
</ul>
</div>


<div id="div10">
<li>The Million Dollar Question</li>
<b>Will I use this spell-checker for the next big revolutionary Word Processing Software?</b>
Let's discuss some advantages of these models:
<ul>
<li>It can detect spelling errors</li>
<li>For context-based models, in many situations, it highlights error words which aren't error but hinting towards a grammatical mistake or a missing filler word.</li> 
Now, what are some problems:
<li>It produces False-Positives.<i>(mostly prevalent in the context-based approach)</i> </li>
<li>The one-hot model classifies the first word mostly as Positive</li>
<li>IT doesn't consider punctuations.</li>
</ul>
What can be done to make it better:
<ul>
<li>Train on a more bigger dataset.</li>
<li>Expand the alphabet to wisely include punctuations as part of the context.</li>
</ul>
</div>



</ol>

