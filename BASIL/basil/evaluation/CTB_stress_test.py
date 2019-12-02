from basil.preprocessing.prep_pytorch_CTB import PytorchPrepWordLevel
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
        #self.prep_torch = PytorchPrepWordLevelWithSyllable(corpus.train_data, corpus.val_data, corpus.test_data)
        # self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
        #                    len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),2,\
        #                    len(self.prep_torch.pos_to_index),len(self.prep_torch.pos2_to_index)).to(device)
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.pos2_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.test_set = corpus.test_data
        self.stress10 = pickle.load( open( "corpora/CTB/CTB10.p", "rb" ) )
        self.stress20 = pickle.load( open( "corpora/CTB/CTB20.p", "rb" ) )
        self.stress30 = pickle.load( open( "corpora/CTB/CTB30.p", "rb" ) )
        self.stress40 = pickle.load( open( "corpora/CTB/CTB40.p", "rb" ) )
        self.stress50 = pickle.load( open( "corpora/CTB/CTB50.p", "rb" ) )
        self.stress60 = pickle.load( open( "corpora/CTB/CTB60.p", "rb" ) )
        self.stress70 = pickle.load( open( "corpora/CTB/CTB70.p", "rb" ) )
        self.stress80 = pickle.load( open( "corpora/CTB/CTB80.p", "rb" ) )
        self.stress90 = pickle.load( open( "corpora/CTB/CTB90.p", "rb" ) )
        self.stress100 = pickle.load( open( "corpora/CTB/CTB100.p", "rb" ) )
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_pos2 = [self.prep_torch.pos2_to_index[word] for text in self.prep_torch.pos2_test for word in text]

    def do_pred(self, stress_set):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_pos2 = []
            for text in stress_set:
                text = [word[0] for word in text]
                wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)
                # pred_pos, pred_pos2 = self.model(wtest_in,ctest_in,ptest_in)
                if len(wtest_in)>0:
                    pred_pos, pred_pos2 = self.model(wtest_in,ctest_in)
                    for p_pos, p_pos2 in  zip(pred_pos, pred_pos2):
                                p_pos= p_pos.data.tolist()
                                p_pos2 = p_pos2.data.tolist()

                                y_pred_pos.append(p_pos.index(max(p_pos)))
                                y_pred_pos2.append(p_pos2.index(max(p_pos2)))
            return y_pred_pos, y_pred_pos2

    def f1_summary(self):
        #tags to include in calculation/evaluation
        pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        # pos_labels.remove(self.prep_torch.pos_to_index['space'])

        pos2_labels = list(range(len(self.prep_torch.index_to_pos2)))
        # pos2_labels.remove(self.prep_torch.pos2_to_index['space'])
        # pos2_labels.remove(self.prep_torch.pos2_to_index['O'])

        y_pred_pos, y_pred_pos2= self.do_pred(self.test_set)
        y_pred_pos10, y_pred_pos210= self.do_pred(self.stress10)
        y_pred_pos20, y_pred_pos220= self.do_pred(self.stress20)
        y_pred_pos30, y_pred_pos230= self.do_pred(self.stress30)
        y_pred_pos40, y_pred_pos240= self.do_pred(self.stress40)
        y_pred_pos50, y_pred_pos250= self.do_pred(self.stress50)
        y_pred_pos60, y_pred_pos260= self.do_pred(self.stress60)
        y_pred_pos70, y_pred_pos270= self.do_pred(self.stress70)
        y_pred_pos80, y_pred_pos280= self.do_pred(self.stress80)
        y_pred_pos90, y_pred_pos290= self.do_pred(self.stress90)
        y_pred_pos100, y_pred_pos2100= self.do_pred(self.stress100)

        f1_pos = f1_score(self.y_true_pos,y_pred_pos, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos2, average='micro',labels=pos2_labels)
        print("normal",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos10, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos210, average='micro',labels=pos2_labels)
        print("10",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos20, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos220, average='micro',labels=pos2_labels)
        print("20",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos30, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos230, average='micro',labels=pos2_labels)
        print("30",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos40, average='micro', labels=pos_labels)

        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos240, average='micro',labels=pos2_labels)
        print("40",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos50, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos250, average='micro',labels=pos2_labels)
        print("50",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos60, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos260, average='micro',labels=pos2_labels)
        print("60",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos70, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos270, average='micro',labels=pos2_labels)
        print("70",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos80, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos280, average='micro',labels=pos2_labels)
        print("80",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos90, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos290, average='micro',labels=pos2_labels)
        print("90",f1_pos,f1_pos2)
        f1_pos = f1_score(self.y_true_pos,y_pred_pos100, average='micro', labels=pos_labels)
        f1_pos2 = f1_score(self.y_true_pos2,y_pred_pos2100, average='micro',labels=pos2_labels)
        print("100",f1_pos,f1_pos2)
