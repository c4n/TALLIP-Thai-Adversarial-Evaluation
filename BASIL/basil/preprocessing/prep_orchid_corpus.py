import re
import glob
import random
from sklearn.model_selection import train_test_split
import yaml
import os

orchid_sym_dict = {'<ampersand>':"&",
                     '<apostrophe>':"'",
                     '<asterisk>':"*",
                     '<at_mark>':"@",
                     '<at_mark>FIXN':"@",
                     '<at_mark>NCMN':"@",
                     '<at_mark>PUNC':"@",
                     '<circumflex_accent>':"^",# NEED REVIEW
                     '<colon>':":",
                     '<comma>':",",
                     '<dollar>':"?",
                     '<equal>':"=",
                     '<exclamation>':"!",
                     '<full_stop>':".",
                     '<greater_than>':">",
                     '<left_curly_bracket>':"{",
                     '<left_parenthesis>':"(",
                     '<less_than>':"<",
                     '<minus>':"-",
                    # '<number>',
                     '<plus>':"+",
                     '<question_mark>':"?",
                     '<quotation>':"\"",
                     '<right_parenthesis>': ")",
                     '<semi_colon>' :";",
                     '<slash>':"/",
                     "<slash>'":"/",
                     '<space>':" "}


class ORCHID:
    'ORCHID corpus preprocessor'
    #note:2 underscores for private variable/method

    def __init__(self,level="word"):
        print("Preprocessing ORCHID corpus")
        if(level=="character"):
            print("coming soon!")

        elif(level=="word"):
            file = open('orchid97.crp.utf', 'r')
            full_text = file.read()
            #(?<=^test_).+(?=\.py$)
            text_files=re.split(r'%File:',full_text)
            file.close()
            text_files=text_files[1:]
            self.clean_dataset = self.__word_level_processor(text_files)
            self.train_data , self.test_data = train_test_split(self.clean_dataset,  test_size=0.10,random_state=42)
            self.train_data , self.val_data = train_test_split(self.train_data,  test_size=0.10,random_state=42)
        else:
            raise ValueError("level can only be 'character' or 'word'")
        print("Done!")


    def __word_level_processor(self,text_list):
        """split words and turn them into a (train/dev/test)set"""

        corpus = []
        for file in text_list:
            temp_text = []
            for line in file.splitlines():
                if len(line)>0:
                        if line[0]!="#" and line[0]!="%" and "//" not in line  and "\\" not in line:
                            this_line = line.split("/")
                            if len(this_line)>1:
                                word,pos = this_line
                                if word in orchid_sym_dict:
                                    word = orchid_sym_dict[word]
                                temp_text.append([word,pos])
            corpus.append(temp_text)


        return corpus
