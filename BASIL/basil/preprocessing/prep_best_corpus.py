import re
import glob
import random
from sklearn.model_selection import train_test_split
import yaml
import os

with open("basil/preprocessing/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

class Best2010:
    'BEST2010(free) corpus preprocessor'
    #note:2 underscores for private variable/method
    __filepath_list = glob.glob(cfg["Best2010"]["file_path"])

    #split into train(appx.80%) ,val(appx.10%) and testset (appx.10%)
    __train_filepath_list = __filepath_list[:int(len(__filepath_list)*0.8)]
    __val_filepath_list = __filepath_list[int(len(__filepath_list)*0.8):int(len(__filepath_list)*0.9)]
    __test_filepath_list = __filepath_list[int(len(__filepath_list)*0.9):]

    def __init__(self,level="character"):
        print("Preprocessing BEST2010 corpus")

        if(level=="character"):
            self.train_data = self.__char_level_processor(self.__train_filepath_list)
            self.val_data = self.__char_level_processor(self.__val_filepath_list)
            self.test_data = self.__char_level_processor(self.__test_filepath_list)

        elif(level=="word"):
            self.train_data = self.__word_level_processor(self.__train_filepath_list)
            self.val_data = self.__word_level_processor(self.__val_filepath_list)
            self.test_data = self.__word_level_processor(self.__test_filepath_list)
        else:
            raise ValueError("level can only be 'character' or 'word'")
        print("Done!")

    def __char_level_processor(self,filepath_list):
        dataset = list()
        for filepath in filepath_list:
            f = open(filepath,"r") #open file with name of "*.txt"
            text = re.sub(r'<\W?\w+>', '', f.read())# remove <NE> </NE> <AB> </AB> tags
            text=re.sub('\\ufeff','',text)#clean out this string,
            #split a text file into lines, since there's no clear sentence boundary
            lines=text.split("\n")

            for line in lines:
                temp=list()
                for word in line.split("|"):
                    B_FLAG=True
                    #loop through each char in a word
                    for char in word:
                        if B_FLAG:
                            temp.append((char,"B"))
                            B_FLAG=False
                        else:
                            temp.append((char,"I"))
                dataset.append(temp)
            f.close()
        return dataset

    def __word_level_processor(self,filepath_list):
        dataset = list()
        for filepath in filepath_list:
            f = open(filepath,"r") #open file with name of "*.txt"
            text=re.sub('\\ufeff','',f.read())#clean out this string,
            #split a text file into lines, since there's no clear sentence boundary
            lines=text.split("\n")
            for line in lines:
                temp = list()
                for word in line.split("|"):
                    word_tag = re.match(r'(<\W?\w+>)', word)
                    word = re.sub(r'<\W?\w+>', '', word)
                    if(word_tag):
                        temp.append((word,word_tag.group()))
                    else:
                        temp.append((word,"O"))
                dataset.append(temp)
            f.close()
        return dataset

class Best2010NECTEC:
    'BEST2010 corpus (internal) preprocessor'
    #note:2 underscores for private variable/method
    __filepath_list = glob.glob(cfg["Best2010NECTEC"]["train_file_path"])
    __train_filepath_list , __extra_val_filepath_list = train_test_split(__filepath_list,  test_size=0.05,random_state=42)
    __val_filepath_list = glob.glob(cfg["Best2010NECTEC"]["val_file_path"])
    __val_filepath_list.extend(__extra_val_filepath_list)
    __test_filepath_list = glob.glob(cfg["Best2010NECTEC"]["test_file_path"])

    def __init__(self,level="word"):
        print("Preprocessing BEST2010 corpus")
        if(level=="character"):
            print("coming soon!")

        elif(level=="word"):
            self.train_data = self.__word_level_processor(self.__train_filepath_list)
            self.val_data = self.__word_level_processor(self.__val_filepath_list)
            self.test_data = self.__word_level_processor(self.__test_filepath_list)
        else:
            raise ValueError("level can only be 'character' or 'word'")
        print("Done!")

    def __cleaner(self,word_rawtext):
        """clean known mistakes after split text into words"""
        word_rawtext=re.sub('/NN//', '/NN/', word_rawtext)
        word_rawtext=re.sub('//PU/', '/PU/', word_rawtext)
        word_rawtext=re.sub('MEA_BI', 'MEA_B', word_rawtext)
        return word_rawtext


    def __word_level_processor(self,filepath_list):
        """split words and turn them into a (train/dev/test)set"""
        the_set = list()
        for filename in filepath_list:
            file = open(filename, 'r',encoding="utf-8",errors='ignore') #unicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 22: invalid start byte
            text=file.read()
            this_text = list()

            for word in text.split('|'):
                if len(word)>1:#if the word is tagged
                    word = self.__cleaner(word)
                    word =  re.findall(r'(.*)/(.*)/(.*)',word)

                    if len(word)<1:#if this pattern is not found it's just a bunch of \n

                        word = ['\n','space','space']# replace multiple \n with just one \n
                    else:
                        word = word[0]
                else:
                    #they are just empty space
                    word = [word,'space','space']

                if len(word[0])>=1:
                    word = list(word)
                    word[1]=word[1].strip()# clean white space around POS tag
                    word[2]=word[2].strip()# clean white space around NER tag
                    this_text.append(word)
                else:
                    #Add space when the space is missing between two ||
                    this_text.append([' ','space','space'])

            the_set.append(this_text)

        return the_set


class BestSyllable2010NECTEC:
    'BEST2010 corpus (internal) preprocessor'
    #note:2 underscores for private variable/method
    __filepath_list = glob.glob(cfg["BestSyllable2010NECTEC"]["train_file_path"])
    __train_filepath_list , __extra_val_filepath_list = train_test_split(__filepath_list,  test_size=0.05,random_state=42)
    __val_filepath_list = glob.glob(cfg["BestSyllable2010NECTEC"]["val_file_path"])
    __val_filepath_list.extend(__extra_val_filepath_list)
    __test_filepath_list = glob.glob(cfg["BestSyllable2010NECTEC"]["test_file_path"])

    def __init__(self,level="word"):
        print("Preprocessing BEST2010 corpus")
        if(level=="character"):
            print("coming soon!")

        elif(level=="word"):
            self.train_data = self.__word_level_processor(self.__train_filepath_list)
            self.val_data = self.__word_level_processor(self.__val_filepath_list)
            self.test_data = self.__word_level_processor(self.__test_filepath_list)
        else:
            raise ValueError("level can only be 'character' or 'word'")
        print("Done!")

    def __cleaner(self,word_rawtext):
        """clean known mistakes after split text into words"""
        word_rawtext=re.sub('/NN//', '/NN/', word_rawtext)
        word_rawtext=re.sub('//PU/', '/PU/', word_rawtext)
        word_rawtext=re.sub('MEA_BI', 'MEA_B', word_rawtext)
        return word_rawtext


    def __word_level_processor(self,filepath_list):
        """split words and turn them into a (train/dev/test)set"""
        the_set = list()
        for filename in filepath_list:
            file = open(filename, 'r',encoding="utf-8",errors='ignore') #unicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 22: invalid start byte
            text=file.read()
            this_text = list()

            for word in text.split('|'):
                if len(word)>1:#if the word is tagged
                    word = self.__cleaner(word)
                    word =  re.findall(r'(.*)/(.*)/(.*)',word)

                    if len(word)<1:#if this pattern is not found it's just a bunch of \n

                        word = ['\n','space','space']# replace multiple \n with just one \n
                    else:
                        word = word[0]
                else:
                    #they are just empty space
                    word = [word,'space','space']

                if len(word[0])>=1:
                    word = list(word)
                    just_word = re.sub(r'~', '', word[0])#remove syllable segment symbol
                    syllable = word[0].split('~')
                    pos=word[1].strip()# clean white space around POS tag
                    ner=word[2].strip()# clean white space around NER tag
                    this_text.append([just_word ,syllable ,pos ,ner])
                else:
                    #Add space when the space is missing between two ||
                    this_text.append([' ',[' '],'space','space'])

            the_set.append(this_text)

        return the_set
