from basil.preprocessing.prep_pytorch import PytorchPrepWordLevel
from basil.evaluation.noise import NoiseGenerator
from sklearn.metrics import f1_score
import torch
import pickle
from basil.select_device import device
from allennlp.modules.elmo import Elmo, batch_to_ids

class SpeedEvalElmo:
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
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)    
        
#         self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
#                            len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),2,\
#                            len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        # self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
        #                    len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
        #                    len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.noise = NoiseGenerator(corpus.test_data)
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_test for word in text]

    def do_pred(self, speed_testset):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_ner = []

            text = [word[0] for word in speed_testset]
#                 wtest_in = self.prep_torch.prepare_sequence_test_word(text)
            wtest_in = batch_to_ids([text]).to(device)
            ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
            ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)

#                 pred_pos, pred_ner = self.model(wtest_in,ctest_in,ptest_in)
            #pred_pos, pred_ner = self.model(wtest_in,ctest_in)
            pred_pos, pred_ner = self.model(wtest_in)
            for p_pos, p_ner in  zip(pred_pos, pred_ner):

                        p_pos= p_pos.data.tolist()
                        p_ner = p_ner.data.tolist()


                        y_pred_pos.append(p_pos.index(max(p_pos)))
                        y_pred_ner.append(p_ner.index(max(p_ner)))
            return y_pred_pos, y_pred_ner
        
        
class SpeedEvalU:
    """stress test"""
    def __init__(self, model, model_weights, corpus):
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        SYL_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128

        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
#         self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
#                            len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
#                            len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)    
        
#         self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
#                            len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),2,\
#                            len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index),54521,\
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.noise = NoiseGenerator(corpus.test_data)
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_test for word in text]

    def do_pred(self, speed_testset):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_ner = []

            text = [word[0] for word in speed_testset]
            wtest_in = self.prep_torch.prepare_sequence_test_word(text)
#             wtest_in = batch_to_ids([text]).to(device)
            ctest_in = []
#             ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)

#                 pred_pos, pred_ner = self.model(wtest_in,ctest_in,ptest_in)
            pred_pos, pred_ner = self.model(wtest_in,ctest_in)
#             pred_pos, pred_ner = self.model(wtest_in)
            for p_pos, p_ner in  zip(pred_pos, pred_ner):

                        p_pos= p_pos.data.tolist()
                        p_ner = p_ner.data.tolist()


                        y_pred_pos.append(p_pos.index(max(p_pos)))
                        y_pred_ner.append(p_ner.index(max(p_ner)))
            return y_pred_pos, y_pred_ner
        
        
class SpeedEvalUBC:
    """stress test"""
    def __init__(self, model, model_weights, corpus):
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        SYL_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128

        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
#         self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
#                            len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
#                            len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)    
        
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),2,\
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
#         self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
#                            len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
#                            len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.noise = NoiseGenerator(corpus.test_data)
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_test for word in text]

    def do_pred(self, speed_testset):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_ner = []

            text = [word[0] for word in speed_testset]
            wtest_in = self.prep_torch.prepare_sequence_test_word(text)
#             wtest_in = batch_to_ids([text]).to(device)
            ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
            ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)

            pred_pos, pred_ner = self.model(wtest_in,ctest_in,ptest_in)
#             pred_pos, pred_ner = self.model(wtest_in,ctest_in)
#             pred_pos, pred_ner = self.model(wtest_in)
            for p_pos, p_ner in  zip(pred_pos, pred_ner):

                        p_pos= p_pos.data.tolist()
                        p_ner = p_ner.data.tolist()


                        y_pred_pos.append(p_pos.index(max(p_pos)))
                        y_pred_ner.append(p_ner.index(max(p_ner)))
            return y_pred_pos, y_pred_ner
        
class SpeedEvalUBCAD:
    """stress test"""
    def __init__(self, model, model_weights, corpus):
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128


        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           184,54521, self.prep_torch.prefix_size,\
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.ner_to_index)).to(device)
        self.model.load_state_dict(torch.load(model_weights))
        self.noise = NoiseGenerator(corpus.test_data)
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_test for word in text]
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_test for word in text]

    def do_pred(self, speed_testset):
        with torch.no_grad():
            y_pred_pos = []
            y_pred_ner = []

            text = [word[0] for word in speed_testset]
            wtest_in = self.prep_torch.prepare_sequence_test_word(text)
#             wtest_in = batch_to_ids([text]).to(device)
            ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
            ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)

            pred_pos, pred_ner = self.model(wtest_in,ctest_in,ptest_in)
#             pred_pos, pred_ner = self.model(wtest_in,ctest_in)
#             pred_pos, pred_ner = self.model(wtest_in)
            for p_pos, p_ner in  zip(pred_pos, pred_ner):

                        p_pos= p_pos.data.tolist()
                        p_ner = p_ner.data.tolist()


                        y_pred_pos.append(p_pos.index(max(p_pos)))
                        y_pred_ner.append(p_ner.index(max(p_ner)))
            return y_pred_pos, y_pred_ner
                