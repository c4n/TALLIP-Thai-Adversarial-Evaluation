__author__ = "Can Udomcharoenchaikit"
__email__ = "udomc.can@gmail.com"
__status__ = "Production"

import random
import re
### ADDITIONAL FEATURES
consonant="กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮ"
final_consonant="กขคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษวฬอ"
number="0123456789๑๒๓๔๕๖๗๘๙"
front_vowel="เแโใไ"
lower_vowel="อุอู".replace('อ','') #ทำแบบนี้จะได้อ่านออก
rear_vowel = "าําๅๆะฯๅๆ"
upper_vowel = "อ็อ้อ์อิอีอือึอํอัอ่อ๋อ๊".replace('อ','')
tone = "อ้อ่อ๋อ๊".replace('อ','')
#regex rules
regex_rule_tone = r"[" + tone +"]"
regex_rule_cons = r"[" + consonant +"]"
regex_rule_fcons = r"[" + final_consonant +"]"
regex_rule_lvowel = r"[" + lower_vowel +"]"
regex_rule_fvowel = r"[" + front_vowel +"]"
regex_rule_uvowel = r"[" + upper_vowel +"]"
regex_rule_ulvowel = r"[" + upper_vowel+lower_vowel +"]"
regex_rule_move_tone = regex_rule_cons+regex_rule_cons+regex_rule_tone


class NoiseGenerator:
    """Generate noisy test set"""
    def __init__(self,test_set):
        self.test_set = test_set
        self.stress10, self.error10 = self.generate_stress_test_set(test_set,0.1)
        self.stress20, self.error20 = self.generate_stress_test_set(test_set,0.2)
        self.stress30, self.error30 = self.generate_stress_test_set(test_set,0.3)
        self.stress40, self.error40 = self.generate_stress_test_set(test_set,0.4)
        self.stress50, self.error50 = self.generate_stress_test_set(test_set,0.5)
        self.stress60, self.error60 = self.generate_stress_test_set(test_set,0.6)
        self.stress70, self.error70 = self.generate_stress_test_set(test_set,0.7)
        self.stress80, self.error80 = self.generate_stress_test_set(test_set,0.8)
        self.stress90, self.error90 = self.generate_stress_test_set(test_set,0.9)
        self.stress100, self.error100 = self.generate_stress_test_set(test_set,1.0)

    def remove_tone(self, in_word):
        out=re.sub(regex_rule_tone,'',in_word)
        return out, True

    def remove_last_char(self, in_word):
        out = in_word
        flag = False
        if len(in_word)>2:
            out=in_word[:-1]
            flag = True
        return out, flag

    def swap_char(self, in_word):
        #swap if one of the last two char is a consonant
        flag = False
        out = in_word
        if bool(re.search(regex_rule_cons,in_word[-2:])) and len(in_word)>2:
            out=''.join([in_word[:-2],in_word[-1],in_word[-2]])
            flag = True
        return out, flag

    def move_tone(self, in_word):
    #move first tone found a bit closer
        out = in_word
        flag = False
        x=re.search(regex_rule_move_tone,in_word)
        if bool(x):
            out=''.join([in_word[:x.start()],\
                        in_word[x.start():x.start()+1],\
                        in_word[x.end()-1],\
                        in_word[x.end()-2],\
                        in_word[x.end():]])
            flag = True
        return out, flag

    def add_stress(self,in_word,stress:float):
        """stress should be between 0 and 1"""
        out = in_word
        error_type = "no_error"
        if random.uniform(0, 1) < stress:
            is_tone = re.search(regex_rule_move_tone,in_word)
            if bool(is_tone)==False:
                error_fnc = [self.remove_last_char,self.swap_char]
                error_types = ['last_char','swap_char']
                choice = random.choice([0,1])
                out, flag  = error_fnc[choice](in_word)
                if flag:
                    error_type = error_types[choice]
                
            else:
                error_fnc = [self.remove_tone,self.move_tone,self.remove_last_char,self.swap_char]
                error_types = ['remove_tone','move_tone','last_char','swap_char']
                choice = random.choice([0,1,2,3])
                out,flag  = error_fnc[choice](in_word)
                if flag:
                    error_type = error_types[choice]                
        return out,error_type


    def generate_stress_test_set(self,test_set,stress):
        random.seed(42)
        stress_set = []
        error_set = []
        for text in test_set:
            stress_text = []
            error_text = []
            for word in text:
                stress_word, stress_type = self.add_stress(word[0],stress)
                stress_text.append([stress_word,word[1],word[2]])
                error_text.append(stress_type)
            stress_set.append(stress_text)
            error_set.append(error_text)
        return stress_set, error_set


class OrchidNoiseGenerator(NoiseGenerator):
    """Generate noisy test set"""
    def __init__(self,test_set):
        super().__init__(test_set)

    def generate_stress_test_set(self,test_set,stress):
        random.seed(42)
        stress_set = []
        error_set = []
        for text in test_set:
            stress_text = []
            error_text = []
            for word in text:
                stress_word, stress_type = self.add_stress(word[0],stress)
                stress_text.append([stress_word,word[1]])
                error_text.append(stress_type)
            stress_set.append(stress_text)
            error_set.append(error_text)
        return stress_set, error_set


class NoiseGeneratorWithSyllable:
    """Generate noisy test set with syllable segmentation"""
    def __init__(self,test_set):
        self.test_set = test_set
        self.stress10 = self.generate_stress_syllable_test_set(test_set,0.1)
        # self.stress20 = self.generate_stress_syllable_test_set(test_set,0.2)
        # self.stress30 = self.generate_stress_syllable_test_set(test_set,0.3)
        # self.stress40 = self.generate_stress_syllable_test_set(test_set,0.4)
        # self.stress50 = self.generate_stress_syllable_test_set(test_set,0.5)
        # self.stress60 = self.generate_stress_syllable_test_set(test_set,0.6)
        # self.stress70 = self.generate_stress_syllable_test_set(test_set,0.7)
        # self.stress80 = self.generate_stress_syllable_test_set(test_set,0.8)
        # self.stress90 = self.generate_stress_syllable_test_set(test_set,0.9)
        # self.stress100 = self.generate_stress_syllable_test_set(test_set,1.0)

    def remove_tone(self, in_word):
        out=re.sub(regex_rule_tone,'',in_word)
        return out

    def remove_last_char(self, in_word):
        out = in_word
        if len(in_word)>2:
            out=in_word[:-1]
        return out

    def swap_char(self, in_word):
        #swap last two consonants
        out = in_word
        if bool(re.search(regex_rule_cons,in_word[-2:])) and len(in_word)>2:
            out=''.join([in_word[:-2],in_word[-1],in_word[-2]])
        return out

    def move_tone(self, in_word):
    #move first tone found a bit closer
        out = in_word
        x=re.search(regex_rule_move_tone,in_word)
        if bool(x):
            out=''.join([in_word[:x.start()],\
                        in_word[x.start():x.start()+1],\
                        in_word[x.end()-1],\
                        in_word[x.end()-2],\
                        in_word[x.end():]])
        return out

    def add_stress(self,in_word,stress:float):
        """stress should be between 0 and 1"""
        out = in_word
        if random.uniform(0, 1) < stress:
            is_tone = re.search(regex_rule_move_tone,in_word)
            if bool(is_tone)==False:
                out = random.choice([self.remove_last_char(in_word),self.swap_char(in_word)])
            else:
                out = random.choice([self.remove_tone(in_word),self.move_tone(in_word),self.remove_last_char(in_word),self.swap_char(in_word)])
        return out


    def generate_stress_syllable_test_set(self,test_set,stress):
        random.seed(42)
        stress_set = []
        for text in test_set:
            stress_text = []
            for word in text:
                stress_word = self.add_stress(word[0],stress)
                stress_syl = tltk.nlp.syl_segment(stress_word)[:-5]
                stress_syl = stress_syl.split("~")
                stress_text.append([stress_word,stress_syl,word[2],word[3]])
            stress_set.append(stress_text)
        return stress_set
