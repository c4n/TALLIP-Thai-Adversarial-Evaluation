import re
import glob
import random
from sklearn.model_selection import train_test_split
import yaml
import os

class CTB:
    'CTB corpus preprocessor'
    #note:2 underscores for private variable/method

    def __init__(self,level="word"):
        print("Preprocessing ORCHID corpus")
        if(level=="character"):
            print("coming soon!")

        elif(level=="word"):
            self.clean_dataset = self.__word_level_processor("corpora/CTB/chinese_sinica_train.conll")
            self.train_data , self.val_data = train_test_split(self.clean_dataset,  test_size=0.10,random_state=42)
            self.test_data = self.__word_level_processor("corpora/CTB/chinese_sinica_test.conll")
        else:
            raise ValueError("level can only be 'character' or 'word'")
        print("Done!")


    def __word_level_processor(self,filename):
        """split words and turn them into a (train/dev/test)set"""
        file = open(filename, 'r')
        full_text = file.read()
        #(?<=^test_).+(?=\.py$)
        text_files=re.split('\n\n',full_text)
        file.close()
        clean_sent = []
        for sent in text_files:
            temp_sent = []
            for word in sent.splitlines():
                row = word.split("\t")
                word = row[1]
                pos1 = row[3]
                pos2 = row[4]
                temp_sent.append([word,pos1,pos2])
            if len(temp_sent) >=1:
                clean_sent.append(temp_sent)


        return clean_sent
