#https://www.geeksforgeeks.org/find-k-frequent-words-data-set-python/
#records in training file : 6232723
#records in development file:10000
#records in test file:10000
import gc

from nltk.corpus import stopwords
from collections import Counter
import json

def get_wikipedia_files(file_path, max_file_lines):
    data=[]
    i=0
    with open(file_path, encoding="utf-8") as infile:
        for i,line in enumerate(infile):
            data.append(json.loads(line)['text'].lower().split())
            if i%10000==0:
                print(f"Line {i} processed")
            if i == max_file_lines:
                return data
    print("File Read complete...")
    return data

def count_words(data, k, max_file_lines):

    flat_list = [item for sublist in data for item in sublist]
    print("Flat List generated")
    del data
    gc.collect()

    stopwords_list = list(stopwords.words('english'))
    c = Counter(flat_list)
    del flat_list
    gc.collect()
    print("Counter generated")

    for x in stopwords_list:
        try:
            c.pop(x)
        except:
            #print("unable to remove",x)
            pass
    print("Stopwords Removed...")
    most_occur = c.most_common()
    most_occur = remove_nonalpha_words(c)
    #dict_d1 = dict(most_occur)#
    #print("dict for also",dict_d1['also'])
    #c2 = Counter(dict_d1)
    #print("len c2",len(c2))#185606
    #most_occur2 = c2.most_common()
    x = json.dumps(dict(most_occur.most_common()), indent=4)
    #print(x)
    json_file = open("top_"+str(k)+"_words_over_"+str(max_file_lines)+".json","w")
    json_file.write(x)
    json_file.close()
    return

def remove_nonalpha_words_flatlist(data):

    for u in data:
        try:
            if u[-1] == ',' or u[-1]=='.':
                #print(u)
                data.append(u[:-1])
                data.remove(u)
                #print("Removed", u," Replaced",u[:-1])
        except:
            print("Cleansing problem for", u)

    for u in data:
        if u.isalpha()==False:
            #print("Removed",u)
            data.remove(u)


def remove_nonalpha_words(data):

    print("Counter length before=", len(data))
    temp = []
    for i,(u,v) in enumerate(data.items()):
        #print(u,v)
        try:
            if u[-1] == ',' or u[-1]=='.':
                #print(u)
                #data.append((u[:-1],v))
                #del data[i]
                temp.append({u[:-1]:v})
        except:
            print("Cleansing problem for",u)
    for x in temp:
        data.update(x)
    print("Counter length between=", len(data))
    removal_list = []
    for i,u in enumerate(data):
        if u=='1900':
            print('gronewald')
        if u.isalpha()==False:
            #print("Removed",u)
            #del data[u]
            removal_list.append(u)

    for x in removal_list:
        del data[x]

    print("Counter length after=", len(data))
    return data


def main ():
    max_file_lines = 100000
    k = "all"
    file_path_dev = "D:\Freiburg\MasterProject\datasets\wikipedia_2021-02-20\\development_documents.jsonl"
    file_path_prod= "D:\Freiburg\MasterProject\datasets\wikipedia_2021-02-20\\training_documents.jsonl"
    file_path_dummy = "D:\\Freiburg\\MasterProject\\data\\3lines_training.jsonl"
    wikipedia_entries = get_wikipedia_files(file_path_dummy , max_file_lines)
    count_words(wikipedia_entries,k, max_file_lines)
    #remove_nonalpha_words([('1900',23),('1900,',25)])
    #print(wikipedia_entries)

if __name__ == "__main__":
    main()