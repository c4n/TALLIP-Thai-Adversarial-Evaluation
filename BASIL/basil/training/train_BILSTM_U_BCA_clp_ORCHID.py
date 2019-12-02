__author__ = "Can Udomcharoenchaikit"
__email__ = "udomc.can@gmail.com"
__status__ = "Production"

from basil.preprocessing.prep_pytorch_ORCHID import PytorchPrepWordLevel, word_type
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score
import time, math
import sys
import random
import torch
from torch import nn
import numpy as np
from basil.select_device import device
noise_level = 0.1
class TrainModel:
    """Train a model given model choice and training set/dev set"""
    def __init__(self,model,corpus,seed_num):
        self.seed_num = seed_num
        np.random.seed(self.seed_num)
        torch.manual_seed(self.seed_num)
        random.seed(self.seed_num)
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128

        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index), self.prep_torch.prefix_size,\
                           len(self.prep_torch.pos_to_index),len(self.prep_torch.type_to_index)).to(device)
        self.loss_function = nn.NLLLoss()
        learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        #for validation process at the end of each epoch
        self.y_true_pos = [self.prep_torch.pos_to_index[word] for text in self.prep_torch.pos_dev for word in text]
        self.y_true_type = [self.prep_torch.type_to_index[word_type[word]] for text in self.prep_torch.pos_dev for word in text]
        #tags to include in calculation/evaluation
        self.pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        self.type_labels = list(range(len(self.prep_torch.index_to_type)))
        #self.pos_labels.remove(self.prep_torch.pos_to_index['space'])
        #print(pos_labels)

    def _generate_unknown_noise(self,text,noise_level):
        "replace word with UNK"
        noisy_text = []
        for word in text:
            noisy_word=word
            if random.uniform(0, 1) < noise_level:
                noisy_word="UNK"
            noisy_text.append(noisy_word)

        return noisy_text

    def _timeSince(self,since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def train(self, epoch_num:int):
        start = time.time()


        print("Training")
        print("Num of training text file:",len(self.prep_torch.input_train_word))
        count=1
        best_f1=0
        noise_level = 0.1
        for epoch in range(epoch_num):


            #switch between normal set and augmented set
        #     if epoch%2==0:
        #         input_text = input_aug_word
        #     else:
        #         input_text = input_train_word
            input_text = self.prep_torch.input_train_word
            for text_file, pos_tags in zip(input_text, self.prep_torch.pos_train):

                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.train()
                self.model.zero_grad()

                # Step 2. Data Prep for pytorch, put them into tensors, chuck them into the GPU!
                #
                text_file=self._generate_unknown_noise(text_file,noise_level)

                word_in = self.prep_torch.prepare_sequence_word(text_file)

                char_in = [ self.prep_torch.prepare_sequence_char(word) for word in text_file]

                prefix_in = self.prep_torch.prepare_sequence_feat_prefix(text_file)
                
                targets_pos =  self.prep_torch.prepare_sequence_target_pos(pos_tags)
                targets_type =  self.prep_torch.prepare_sequence_target_word_type(pos_tags)


                # Step 3. Run our forward pass.
                tag_scores_pos, tag_scores_type = self.model(word_in,char_in,prefix_in)


                #Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss_pos = self.loss_function(tag_scores_pos,  targets_pos).to(device)
                loss_type = self.loss_function(tag_scores_type,  targets_type).to(device)

                loss =  loss_pos + loss_type
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 50)
                self.optimizer.step()

                #Step 5: log just to help analyze
                # Compute accuracy
                #POS
                _, argmax = torch.max(tag_scores_pos, 1)
                accuracy_pos = (targets_pos == argmax.squeeze()).float().mean()
                _, argmax = torch.max(tag_scores_type, 1)
                accuracy_type = (targets_type == argmax.squeeze()).float().mean()


                # 1. Log scalar values (scalar summary)
                info = { 'loss_pos': loss_pos.item(),\
                        'accuracy_pos': accuracy_pos.item(),\
                        'accuracy_type': accuracy_type.item()}

        #         for tag, value in info.items():
        #             logger.scalar_summary(tag, value, count)


                ####PRINT STATUS
                count+=1
                if count % 2 ==0:
                    sys.stdout.write("\r epoch "+str(epoch)+": "+self._timeSince(start)\
                                     +" loss:" + str(loss.item())[:6]\
                                     + " accuracy POS:" + str(accuracy_pos.item())[:6]\
                                     + " accuracy TYPE:" + str(accuracy_type.item())[:6])
                    sys.stdout.flush()


            # ================================================================== #
            #                        End of epoch                                #
            # ================================================================== #

            #####Calculate on validation set and save best model
            with torch.no_grad():
                y_pred_pos = []
                y_pred_type = []
                for text in self.prep_torch.input_dev_word:
                    wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                    ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                    ptest_in = self.prep_torch.prepare_sequence_feat_prefix(text)
                    pred_pos, pred_type = self.model(wtest_in,ctest_in,ptest_in)
                    for p_pos, p_type in zip(pred_pos, pred_type):

                                p_pos= p_pos.data.tolist()
                                p_type= p_type.data.tolist()

                                y_pred_pos.append(p_pos.index(max(p_pos)))
                                y_pred_type.append(p_type.index(max(p_type)))
            f1_pos = f1_score(self.y_true_pos,y_pred_pos, average='micro', labels=self.pos_labels)
            f1_type = f1_score(self.y_true_type,y_pred_type, average='micro', labels=self.type_labels)


            print("epoch:",epoch)
            print("F1 (POS):",f1_pos)
            if f1_pos > best_f1:
                torch.save(self.model.state_dict(), "weights/model_BILSTM_U_BCA_clp_orchid_seed"+str(self.seed_num)+".p")#save best model only
                best_f1 = f1_pos

