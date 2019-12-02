import tltk
import re
import yaml
import os
import glob
#from tqdm import tqdm

with open("basil/preprocessing/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

def create_syllable_folder(directory):
    if not os.path.exists(directory+"_syllable"):
        os.makedirs(directory+"_syllable")

def create_nectec_syllable_seg_file(filepath):
    #for NECTEC corpus

    file = open(filepath, 'r',encoding="utf-8",errors='ignore') #unicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 22: invalid start byte

    text=file.read()
    file.close()
    this_text=list()

    for word in text.split('|'):

        if len(word)>1:#if the word is tagged

            tagged_word =  re.findall(r'(.*)/(.*)/(.*)',word)#[(tuple)]
            if len(tagged_word)<1:#if this pattern is not found it's just a bunch of \n
                this_text.append(word)
            else:
                tagged_word=list(tagged_word[0])#convert tuple to list
                tagged_word[0]=tltk.nlp.syl_segment(tagged_word[0])[:-5]#-5 to ignore <\s> tag and the last ~

                this_text.append("/".join(tagged_word))

        else:
            #they are just empty space
            this_text.append(word)
    this_text="|".join(this_text)
    writefilepath = filepath.split("/")
    writefilepath[-2] += "_syllable"
    writefilepath = "/".join(writefilepath)
    writefile  = open(writefilepath,'w',encoding="utf-8",)
    writefile.write(this_text)
    writefile.close()
    return this_text

def create_syllable_seg_corpus():
    train_filepath_list = glob.glob(cfg["Best2010NECTEC"]["train_file_path"])
    for filepath in tqdm(train_filepath_list):
        create_nectec_syllable_seg_file(filepath)

    val_filepath_list = glob.glob(cfg["Best2010NECTEC"]["val_file_path"])
    for filepath in tqdm(val_filepath_list):
        create_nectec_syllable_seg_file(filepath)

    test_filepath_list = glob.glob(cfg["Best2010NECTEC"]["test_file_path"])
    for filepath in tqdm(test_filepath_list):
        create_nectec_syllable_seg_file(filepath)


def main():
    print("running")
    create_syllable_folder("corpora/BEST2010_I2R/Train")
    create_syllable_folder("corpora/BEST2010_I2R/Dev")
    create_syllable_folder("corpora/BEST2010_I2R/Test")
    #    create_syllable_seg_corpus()
    # train_filepath_list = glob.glob(cfg["Best2010NECTEC"]["train_file_path"])
    # for filepath in tqdm(train_filepath_list):
    #     create_nectec_syllable_seg_file(filepath)

    # val_filepath_list = glob.glob(cfg["Best2010NECTEC"]["val_file_path"])
    # for i in range(len(val_filepath_list)):
    #     print(i,len(val_filepath_list))
    #     create_nectec_syllable_seg_file(val_filepath_list[i])

    test_filepath_list = glob.glob(cfg["Best2010NECTEC"]["test_file_path"])
    for i in range(len(test_filepath_list)):
        print(i,len(test_filepath_list))
        create_nectec_syllable_seg_file(test_filepath_list[i])

if __name__ == "__main__":
    # execute only if run as a script
    main()
