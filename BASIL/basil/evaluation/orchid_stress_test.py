from basil.preprocessing.prep_pytorch_ORCHID import PytorchPrepWordLevel
from basil.evaluation.noise import OrchidNoiseGenerator
from sklearn.metrics import f1_score
import torch
from basil.select_device import device
from allennlp.modules.elmo import Elmo, batch_to_ids
import pickle

class OrchidStressEval:
    """stress test"""
    def __init__(self, model, model_weights, corpus,experiment_name):
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        SYL_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128
        self.experiment_name = experiment_name
        
        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
        self.model = model
        self.model.load_state_dict(torch.load(model_weights))
        self.noise = pickle.load( open( "stress_orchid.p", "rb" ) )
        self.noise = OrchidNoiseGenerator(corpus.test_data)
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]


    def do_pred(self, stress_set,input_type):
        with torch.no_grad():
            y_pred_pos = []

            for text in stress_set:
                text = [word[0] for word in text]
                if input_type == "elmo":
#                 wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                    wtest_in = batch_to_ids([text]).to(device)
                    pred_pos, pred_f = self.model(wtest_in)
                elif input_type == "word":
                    wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                    pred_pos, pred_f = self.model(wtest_in)
                elif input_type == "char":
                    wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                    ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                    pred_pos, pred_f = self.model(wtest_in,ctest_in)      
                elif input_type == "affix":
                    wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                    ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                    ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)                    
                    pred_pos, pred_f = self.model(wtest_in,ctest_in,ptest_in)      
                                        
                for p_pos in  pred_pos:
                    p_pos= p_pos.data.tolist()
                    y_pred_pos.append(p_pos.index(max(p_pos)))

            return y_pred_pos

    def f1_summary(self,input_type):
        #tags to include in calculation/evaluation
        pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        # pos_labels.remove(self.prep_torch.pos_to_index['space'])


        y_pred_pos= self.do_pred(self.noise.test_set,input_type)
        y_pred_pos10= self.do_pred(self.noise.stress10,input_type)
        y_pred_pos20= self.do_pred(self.noise.stress20,input_type)
        y_pred_pos30= self.do_pred(self.noise.stress30,input_type)
        y_pred_pos40= self.do_pred(self.noise.stress40,input_type)
        y_pred_pos50= self.do_pred(self.noise.stress50,input_type)
        y_pred_pos60= self.do_pred(self.noise.stress60,input_type)
        y_pred_pos70= self.do_pred(self.noise.stress70,input_type)
        y_pred_pos80= self.do_pred(self.noise.stress80,input_type)
        y_pred_pos90= self.do_pred(self.noise.stress90,input_type)
        y_pred_pos100= self.do_pred(self.noise.stress100,input_type)

        y_pred_pos_set = [y_pred_pos, y_pred_pos10, y_pred_pos20, y_pred_pos30, y_pred_pos40,
                          y_pred_pos50, y_pred_pos60, y_pred_pos70, y_pred_pos80, y_pred_pos90, y_pred_pos100]
        pickle.dump( y_pred_pos_set, open( "y_pred_pos_set"+self.experiment_name+".p", "wb" ) )
        
        f1_pos = f1_score(self.y_true_pos,y_pred_pos, average='micro', labels=pos_labels)
        print("normal",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos10, average='micro', labels=pos_labels)
        print("10",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos20, average='micro', labels=pos_labels)
        print("20",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos30, average='micro', labels=pos_labels)
        print("30",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos40, average='micro', labels=pos_labels)
        print("40",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos50, average='micro', labels=pos_labels)
        print("50",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos60, average='micro', labels=pos_labels)
        print("60",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos70, average='micro', labels=pos_labels)
        print("70",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos80, average='micro', labels=pos_labels)
        print("80",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos90, average='micro', labels=pos_labels)
        print("90",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos100, average='micro', labels=pos_labels)
        print("100",f1_pos)

    def f1_summary_load(self):
        #tags to include in calculation/evaluation
        pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        # pos_labels.remove(self.prep_torch.pos_to_index['space'])
        y_pred_pos_set=pickle.load( open("y_pred_pos_set"+self.experiment_name+".p", "rb" ) )
        [y_pred_pos, y_pred_pos10, y_pred_pos20, y_pred_pos30, y_pred_pos40,y_pred_pos50, y_pred_pos60, y_pred_pos70, y_pred_pos80, y_pred_pos90, y_pred_pos100] = y_pred_pos_set
        
        f1_pos = f1_score(self.y_true_pos,y_pred_pos, average='micro', labels=pos_labels)
        print("normal",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos10, average='micro', labels=pos_labels)
        print("10",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos20, average='micro', labels=pos_labels)
        print("20",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos30, average='micro', labels=pos_labels)
        print("30",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos40, average='micro', labels=pos_labels)
        print("40",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos50, average='micro', labels=pos_labels)
        print("50",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos60, average='micro', labels=pos_labels)
        print("60",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos70, average='micro', labels=pos_labels)
        print("70",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos80, average='micro', labels=pos_labels)
        print("80",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos90, average='micro', labels=pos_labels)
        print("90",f1_pos)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos100, average='micro', labels=pos_labels)
        print("100",f1_pos)
#if __name__ == '__main__':
# best2010  = Best2010NECTEC()
# stress_eval  = StressEval(LSTMTagger, "model_sep_2018best1_affix.p",best2010 )
# stress_eval.f1_summary()
