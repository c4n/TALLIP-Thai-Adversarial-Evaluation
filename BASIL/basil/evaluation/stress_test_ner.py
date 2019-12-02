from basil.preprocessing.prep_pytorch import PytorchPrepWordLevel
from basil.evaluation.noise import NoiseGenerator
from sklearn.metrics import f1_score
import torch
import pickle
from basil.select_device import device


class StressEval:
    """stress test"""
    def __init__(self, model, model_weights, corpus):
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        SYL_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128

        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
                           len(self.prep_torch.ner_to_index)).to(device)
        # self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
        #                    len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
        #                    len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.noise = NoiseGenerator(corpus.test_data)
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_test for word in text]

    def do_pred(self, stress_set):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_ner = []
            for text in stress_set:
                text = [word[0] for word in text]
                wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)
                pred_ner = self.model(wtest_in,ctest_in)
                # pred_pos, pred_ner = self.model(wtest_in,ctest_in)
                for p_ner in  pred_ner:

                            p_ner= p_ner.data.tolist()

                            y_pred_ner.append(p_ner.index(max(p_ner)))
                            #y_pred_ner.append(p_ner.index(max(p_ner)))
            return y_pred_ner, y_pred_ner

    def f1_summary(self):
        #tags to include in calculation/evaluation
        pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        pos_labels.remove(self.prep_torch.pos_to_index['space'])

        ner_labels = list(range(len(self.prep_torch.index_to_ner)))
        ner_labels.remove(self.prep_torch.ner_to_index['space'])
        ner_labels.remove(self.prep_torch.ner_to_index['O'])

        y_pred_pos, y_pred_ner= self.do_pred(self.noise.test_set)
        y_pred_pos10, y_pred_ner10= self.do_pred(self.noise.stress10)
        y_pred_pos20, y_pred_ner20= self.do_pred(self.noise.stress20)
        y_pred_pos30, y_pred_ner30= self.do_pred(self.noise.stress30)
        y_pred_pos40, y_pred_ner40= self.do_pred(self.noise.stress40)
        y_pred_pos50, y_pred_ner50= self.do_pred(self.noise.stress50)
        y_pred_pos60, y_pred_ner60= self.do_pred(self.noise.stress60)
        y_pred_pos70, y_pred_ner70= self.do_pred(self.noise.stress70)
        y_pred_pos80, y_pred_ner80= self.do_pred(self.noise.stress80)
        y_pred_pos90, y_pred_ner90= self.do_pred(self.noise.stress90)
        y_pred_pos100, y_pred_ner100= self.do_pred(self.noise.stress100)
        f1_pos = "N/A"
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner, average='micro',labels=ner_labels)
        #f1_ner = "N/A"
        print("normal",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos10, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner10, average='micro',labels=ner_labels)
        print("10",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos20, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner20, average='micro',labels=ner_labels)
        print("20",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos30, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner30, average='micro',labels=ner_labels)
        print("30",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos40, average='micro', labels=pos_labels)

        f1_ner = f1_score(self.y_true_ner,y_pred_ner40, average='micro',labels=ner_labels)
        print("40",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos50, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner50, average='micro',labels=ner_labels)
        print("50",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos60, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner60, average='micro',labels=ner_labels)
        print("60",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos70, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner70, average='micro',labels=ner_labels)
        print("70",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos80, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner80, average='micro',labels=ner_labels)
        print("80",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos90, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner90, average='micro',labels=ner_labels)
        print("90",f1_pos,f1_ner)
        #f1_pos = f1_score(self.y_true_pos,y_pred_pos100, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner100, average='micro',labels=ner_labels)
        print("100",f1_pos,f1_ner)



class StressSyllableEval:
    """stress test"""
    def __init__(self, model, model_weights, corpus):
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        SYL_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128

        self.prep_torch = PytorchPrepWordLevelWithSyllable(corpus.train_data, corpus.val_data, corpus.test_data)
        self.model = model(CHAR_EMBEDDING_DIM, SYL_EMBEDDING_DIM, WORD_EMBEDDING_DIM, HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index), len(self.prep_torch.syl_to_index), len(self.prep_torch.word_to_index), self.prep_torch.prefix_size,\
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        # self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
        #                    len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
        #                    len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.test_set = corpus.test_data
        self.stress10 = pickle.load( open( "stress10.p", "rb" ) )
        self.stress20 = pickle.load( open( "stress20.p", "rb" ) )
        self.stress30 = pickle.load( open( "stress30.p", "rb" ) )
        self.stress40 = pickle.load( open( "stress40.p", "rb" ) )
        self.stress50 = pickle.load( open( "stress50.p", "rb" ) )
        self.stress60 = pickle.load( open( "stress60.p", "rb" ) )
        self.stress70 = pickle.load( open( "stress70.p", "rb" ) )
        self.stress80 = pickle.load( open( "stress80.p", "rb" ) )
        self.stress90 = pickle.load( open( "stress90.p", "rb" ) )
        self.stress100 = pickle.load( open( "stress100.p", "rb" ) )

        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_test for word in text]

    def do_pred(self, stress_set):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_ner = []
            for text in stress_set:
                syl_text = [word[1] for word in text]
                text = [word[0] for word in text]
                wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                stest_in = [self.prep_torch.prepare_sequence_test_syl(word) for word in syl_text]
                ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)
                pred_pos, pred_ner = self.model(wtest_in,stest_in,ctest_in,ptest_in)
                # pred_pos, pred_ner = self.model(wtest_in,ctest_in)
                for p_pos, p_ner in  zip(pred_pos, pred_ner):

                            p_pos= p_pos.data.tolist()
                            p_ner = p_ner.data.tolist()


                            y_pred_pos.append(p_pos.index(max(p_pos)))
                            y_pred_ner.append(p_ner.index(max(p_ner)))
            return y_pred_pos, y_pred_ner

    def f1_summary(self):
        #tags to include in calculation/evaluation
        pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        pos_labels.remove(self.prep_torch.pos_to_index['space'])

        ner_labels = list(range(len(self.prep_torch.index_to_ner)))
        ner_labels.remove(self.prep_torch.ner_to_index['space'])
        ner_labels.remove(self.prep_torch.ner_to_index['O'])

        y_pred_pos, y_pred_ner= self.do_pred(self.test_set)
        y_pred_pos10, y_pred_ner10= self.do_pred(self.stress10)
        y_pred_pos20, y_pred_ner20= self.do_pred(self.stress20)
        y_pred_pos30, y_pred_ner30= self.do_pred(self.stress30)
        y_pred_pos40, y_pred_ner40= self.do_pred(self.stress40)
        y_pred_pos50, y_pred_ner50= self.do_pred(self.stress50)
        y_pred_pos60, y_pred_ner60= self.do_pred(self.stress60)
        y_pred_pos70, y_pred_ner70= self.do_pred(self.stress70)
        y_pred_pos80, y_pred_ner80= self.do_pred(self.stress80)
        y_pred_pos90, y_pred_ner90= self.do_pred(self.stress90)
        y_pred_pos100, y_pred_ner100= self.do_pred(self.stress100)

        f1_pos = f1_score(self.y_true_pos,y_pred_pos, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner, average='micro',labels=ner_labels)
        print("normal",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos10, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner10, average='micro',labels=ner_labels)
        print("10",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos20, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner20, average='micro',labels=ner_labels)
        print("20",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos30, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner30, average='micro',labels=ner_labels)
        print("30",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos40, average='micro', labels=pos_labels)

        f1_ner = f1_score(self.y_true_ner,y_pred_ner40, average='micro',labels=ner_labels)
        print("40",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos50, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner50, average='micro',labels=ner_labels)
        print("50",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos60, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner60, average='micro',labels=ner_labels)
        print("60",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos70, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner70, average='micro',labels=ner_labels)
        print("70",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos80, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner80, average='micro',labels=ner_labels)
        print("80",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos90, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner90, average='micro',labels=ner_labels)
        print("90",f1_pos,f1_ner)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos100, average='micro', labels=pos_labels)
        f1_ner = f1_score(self.y_true_ner,y_pred_ner100, average='micro',labels=ner_labels)
        print("100",f1_pos,f1_ner)
#if __name__ == '__main__':
# best2010  = Best2010NECTEC()
# stress_eval  = StressEval(LSTMTagger, "model_sep_2018best1_affix.p",best2010 )
# stress_eval.f1_summary()
