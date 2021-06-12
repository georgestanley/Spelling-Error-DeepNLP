import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords


#blog_data = np.genfromtxt(fname="blog_authorship_corpus/blogtext.csv", encoding="utf8")
blog_data = pd.read_csv("blog_authorship_corpus/blogtext.csv", nrows=100000)
print(blog_data.shape)
print("Head..",blog_data.head())
print("Info..", blog_data.info())
#print(blog_data.loc[5][6])

print(blog_data.gender.value_counts())
print(blog_data.isna().sum())

# remove unwanted chars other than alphanumeric
pattern = "[^\w ]"
blog_data.text = blog_data.text.apply(lambda s : re.sub(pattern,"",s))

#convert to lowercase
blog_data.text = blog_data.text.apply(lambda s: s.lower(), lambda s:s.strip())

#remove unwanted spaces
#blog_data.text = blog_data.text.apply(lambda s: s.strip())

#remove stopwords
stopwords=set(stopwords.words('english'))
#blog_data.text = blog_data.text.apply(lambda t: ' '.join([words for words in t.split() if words not in stopwords]) )

print("Head..",blog_data.head())

# drop id and date columns
#blog_data.drop(labels=['id','date'], axis=1,inplace=True)






