import random
import re
import pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi
from Pinyin2Hanzi import is_pinyin, simplify_pinyin

class NoiseGenerator:
    """Generate noisy test set for Chinese Tree Bank dataset"""
    def __init__(self,test_set):
        self.hmmparams = DefaultHmmParams()
        self.test_set = test_set
        self.stress10 = self.generate_stress_test_set(test_set,0.1)
        self.stress20 = self.generate_stress_test_set(test_set,0.2)
        self.stress30 = self.generate_stress_test_set(test_set,0.3)
        self.stress40 = self.generate_stress_test_set(test_set,0.4)
        self.stress50 = self.generate_stress_test_set(test_set,0.5)
        self.stress60 = self.generate_stress_test_set(test_set,0.6)
        self.stress70 = self.generate_stress_test_set(test_set,0.7)
        self.stress80 = self.generate_stress_test_set(test_set,0.8)
        self.stress90 = self.generate_stress_test_set(test_set,0.9)
        self.stress100 = self.generate_stress_test_set(test_set,1.0)


    def add_stress(self,in_word,stress:float):
        """stress should be between 0 and 1"""
        out = in_word
        if random.uniform(0, 1) < stress and not notChinese(in_word):
            word_pinyin= pinyin.get(in_word,  format="strip",delimiter=" ").split()
            for i in range(len(word_pinyin)):
                if not is_pinyin(word_pinyin[i]):
                    word_pinyin[i]=simplify_pinyin(word_pinyin[i])

            result = viterbi(hmm_params=self.hmmparams, observations=word_pinyin, path_num = 2)
            for item in result:
                if "".join(item.path) != "".join(word_pinyin):
                    out ="".join(item.path)
                    break
        return out

    def notChinese(self,context):
        filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
        context = filtrate.match(context) # match

        return bool(context)


    def generate_stress_test_set(self,test_set,stress):
        random.seed(42)
        stress_set = []
        for text in test_set:
            stress_text = []
            for word in text:
                stress_word = self.add_stress(word[0],stress)
                stress_text.append([stress_word,word[1],word[2]])
            stress_set.append(stress_text)
        return stress_set
