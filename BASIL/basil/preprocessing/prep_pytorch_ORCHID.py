__author__ = "Can Udomcharoenchaikit"
__email__ = "udomc.can@gmail.com"
__status__ = "Production"

import torch
import re
from basil.select_device import device

#function or content word
word_type = {'ADVI': 'C',
             'ADVN': 'C',
             'ADVP': 'C',
             'ADVS': 'C',
             'CFQC': 'C',
             'CLTV': 'C',
             'CMTR': 'C',
             'CNIT': 'C',
             'CVBL': 'C',
             'DCNM': 'C',
             'DDAC': 'F',
             'DDAN': 'F',
             'DDAQ': 'F',
             'DDBQ': 'F',
             'DIAC': 'F',
             'DIAQ': 'F',
             'DIBQ': 'F',
             'DONM': 'C',
             'EAFF': 'F',
             'EITT': 'F',
             'FIXN': 'F',
             'FIXV': 'F',
             'JCMP': 'F',
             'JCRG': 'F',
             'JSBR': 'F',
             'NCMN': 'C',
             'NCNM': 'C',
             'NEG':  'F',
             'NLBL': 'C',
             'NONM': 'C',
             'NPRP': 'C',
             'NTTL': 'C',
             'PDMN': 'F',
             'PNTR': 'F',
             'PPRS': 'F',
             'PREL': 'F',
             'PUNC': 'F',
             'RPRE': 'F',
             'VACT': 'C',
             'VATT': 'C',
             'VSTA': 'C',
             'XVAE': 'F',
             'XVAM': 'F',
             'XVBB': 'F',
             'XVBM': 'F',
             'XVMM': 'F'}
class PytorchPrepWordLevel:
    """Prepare ORCHID data for pytorch"""
    def __init__(self,train_set,dev_set,test_set):
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.word_to_index, self.char_to_index, self.word_count = self.__word2index()

        #Convert dataset into this format (word,POS,NE) or (word,POS,NE,oov)
        self.target_train_set = self._create_target_set(self.train_set)
        self.target_dev_set = self._create_target_set(self.dev_set)
        self.target_test_set = self._create_target_set(self.test_set)
        self.target_test_set_oov = self._create_target_test_set_oov(self.test_set)
        #Generate INPUT and label ready to be preprocess bt Pytorch
        self.input_train_word, self.pos_train = self._split_input_label(self.target_train_set)
        self.input_dev_word, self.pos_dev = self._split_input_label(self.target_dev_set)
        self.input_test_word ,self.pos_test = self._split_input_label(self.target_test_set)
        #Generate pos2index, ne2index
        all_pos = sorted(list(set([val for sublist in self.pos_train+self.pos_dev+self.pos_test for val in sublist])))
        self.pos_to_index = dict((c, i) for i, c in enumerate(all_pos))
        self.type_to_index = {'F':0, 'C':1}


        self.index_to_pos = dict((v,k) for k,v in self.pos_to_index.items())
        self.index_to_type = dict((v,k) for k,v in self.type_to_index.items())

        self.prefix_size = 2 # True or False

        ##REGEX RULES
        self.regex_rule_affixes = r"^การ|^ความ|^นัก|^ผู้|^ไอ้|กร$|^ที่\w+|^อย่าง|^แบบ|^โดย|^น่า"

    def __word2index(self):
        word_list=[]
        raw_text=""
        for text in self.train_set:
            for word in text:

                if len(word)==0:
                    pass
                else:
                    word_list.append(word[0])
                    raw_text+=word[0]

        #count word
        word_count = {}
        for word in word_list:
            word_count[word] = word_count.get(word,0) +1
        word_count["UNK"]=0
        ##all words in train set
        all_words = sorted(set(word_list))
        all_words.append("UNK")
        word_to_index = dict((c, i) for i, c in enumerate(all_words))


        ##char2index
        all_characters = sorted(list(set(raw_text)))
        all_characters.append("UNK")
        del raw_text
        char_to_index = dict((c, i) for i, c in enumerate(all_characters))

        return word_to_index, char_to_index, word_count

    def _create_target_set(self,input_set):
        target_set=[]
        for text in input_set:
            temp=[]
            for word in text:

                if len(word)>1:
                    pos_class=word[1]
                    temp.append((word[0],pos_class))

            target_set.append(temp)#for each text chunk/file
        return target_set

    def _create_target_test_set_oov(self,input_set):
        """target_set with OOV meta information"""
        target_set=[]
        for text in input_set:
            temp=[]
            for word in text:
                oov=False
                if len(word)>1:
                    pos_class=word[1]
                    if word[0] in self.word_to_index:
                        pass
                    else:
                        oov=True
                    temp.append((word[0],pos_class,oov))
            target_set.append(temp)#for each text chunk/file
        return target_set

    def _split_input_label(self,target_set):
        """split into input and label(s) from annotated clean dataset <word,POS,NE>"""
        input_word = []
        input_syl = []
        pos = []
        ner = []
        for text in target_set:
            temp_input=[]
            temp_pos=[]
            temp_ner=[]
            for word in text:
                temp_input.append(word[0])
                temp_pos.append(word[1])
            input_word.append(temp_input)
            pos.append(temp_pos)
        return input_word, pos

    ###Pytorch format
    def prepare_sequence_test_word(self, test_text):
        idxs=list()
        for w in test_text:
            if w in self.word_to_index:
                #REFACTOR THIS LATER..
                if self.word_count[w]>2:
                    idxs.append(self.word_to_index[w])
                else:
                    idxs.append(self.word_to_index["UNK"])
            else:
                idxs.append(self.word_to_index["UNK"])

        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)

    def prepare_sequence_word(self, input_text):
        idxs=list()
        for w in input_text:
            if self.word_count[w]>2:
                idxs.append(self.word_to_index[w])
            else:
                idxs.append(self.word_to_index["UNK"])

        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)


    def prepare_sequence_char(self, input_word):
        idxs = [self.char_to_index[c] for c in input_word]
        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)

    def prepare_sequence_test_char(self, test_word):
        idxs=list()
        for c in test_word:
            if c in self.char_to_index:
                idxs.append(self.char_to_index[c])
            else:
                idxs.append(self.char_to_index["UNK"])
        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)

    def prepare_sequence_target_pos(self, input_label):
        idxs = [self.pos_to_index[w] for w in input_label]
        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)

    def prepare_sequence_target_word_type(self, input_label):
        idxs = [self.type_to_index[word_type[w]] for w in input_label]
        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)

    def prepare_sequence_feat_prefix(self, input_text):
        idxs = []
        for word in input_text:
            if bool(re.search(self.regex_rule_affixes,word)):
                idxs.append(1)
            else:
                idxs.append(0)
        tensor = torch.tensor(idxs, dtype=torch.long)
        return tensor.to(device)
