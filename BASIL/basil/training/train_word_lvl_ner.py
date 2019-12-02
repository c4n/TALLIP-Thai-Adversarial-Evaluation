__author__ = "Can Udomcharoenchaikit"
__email__ = "udomc.can@gmail.com"
__status__ = "Production"

from basil.preprocessing.prep_best_corpus import Best2010NECTEC
from basil.preprocessing.prep_pytorch import PytorchPrepWordLevel
from basil.models.multitasks.wordlvl_lstm import LSTMTagger
from sklearn.metrics import f1_score
import time, math
import sys
import random
import torch
from torch import nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
noise_level = 0.1
class TrainModel:
    """Train a model given model choice and training set/dev set"""
    def __init__(self,model,corpus):
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        #REPLACE hyperparams WITH  A CONFIG FILE
        CHAR_EMBEDDING_DIM = 100
        SYL_EMBEDDING_DIM = 100
        WORD_EMBEDDING_DIM = 128
        HIDDEN_DIM = 128

        self.prep_torch = PytorchPrepWordLevel(corpus.train_data, corpus.val_data, corpus.test_data)
        self.model = model(CHAR_EMBEDDING_DIM,WORD_EMBEDDING_DIM,HIDDEN_DIM,\
                           len(self.prep_torch.char_to_index),len(self.prep_torch.word_to_index),\
                           len(self.prep_torch.ner_to_index)).to(device)
        self.loss_function = nn.NLLLoss()
        learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        #for validation process at the end of each epoch
        self.y_true_ner = [self.prep_torch.ner_to_index[word] for text in self.prep_torch.ner_dev for word in text]
        #tags to include in calculation/evaluation
        self.pos_labels = list(range(len(self.prep_torch.index_to_pos)))
        self.pos_labels.remove(self.prep_torch.pos_to_index['space'])
        #print(pos_labels)
        self.ner_labels = list(range(len(self.prep_torch.index_to_ner)))
        self.ner_labels.remove(self.prep_torch.ner_to_index['space'])
        self.ner_labels.remove(self.prep_torch.ner_to_index['O'])
        #print(ner_labels)

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
        noise_level = 0.1#no UNK dropout #changte to 0.1 for unk dropout
        for epoch in range(epoch_num):


            #switch between normal set and augmented set
        #     if epoch%2==0:
        #         input_text = input_aug_word
        #     else:
        #         input_text = input_train_word
            input_text = self.prep_torch.input_train_word
            for text_file, ner_tags in zip(input_text,self.prep_torch.ner_train):

                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.train()
                self.model.zero_grad()

                # Step 2. Data Prep for pytorch, put them into tensors, chuck them into the GPU!
                #
                text_file=self._generate_unknown_noise(text_file,noise_level)

                word_in = self.prep_torch.prepare_sequence_word(text_file)

                char_in = [ self.prep_torch.prepare_sequence_char(word) for word in text_file]

                targets_ner =  self.prep_torch.prepare_sequence_target_ner(ner_tags)
                #targets_ner =  self.prep_torch.prepare_sequence_target_ner(ner_tags)
                # Step 3. Run our forward pass.
                tag_scores_ner = self.model(word_in,char_in)


                #Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss_ner = self.loss_function(tag_scores_ner,  targets_ner).to(device)
            #    loss_ner = self.loss_function(tag_scores_ner,  targets_ner).to(device)
                loss =  loss_ner
                loss.backward()
                self.optimizer.step()

                #Step 5: log just to help analyze
                # Compute accuracy
                #POS
                _, argmax = torch.max(tag_scores_ner, 1)
                accuracy_ner = (targets_ner == argmax.squeeze()).float().mean()
                #NER
                # _, argmax = torch.max(tag_scores_ner, 1)
                # accuracy_ner = (targets_ner == argmax.squeeze()).float().mean()

                # 1. Log scalar values (scalar summary)
                info = { 'loss_ner': loss_ner.item(),\
                        'accuracy_ner': accuracy_ner.item()}
        #         for tag, value in info.items():
        #             logger.scalar_summary(tag, value, count)


                ####PRINT STATUS
                count+=1
                if count % 2 ==0:
                    sys.stdout.write("\r epoch "+str(epoch)+": "+self._timeSince(start)\
                                     +" loss:" + str(loss.item())[:6]\
                                     + " accuracy ner:" + str(accuracy_ner.item())[:6] )
                    sys.stdout.flush()


            # ================================================================== #
            #                        End of epoch                                #
            # ================================================================== #

            #####Calculate on validation set and save best model
            with torch.no_grad():

                y_pred_ner = []
                for text in self.prep_torch.input_dev_word:
                    wtest_in = self.prep_torch.prepare_sequence_test_word(text)
                    ctest_in = [self.prep_torch.prepare_sequence_test_char(word) for word in text]
                    pred_ner = self.model(wtest_in,ctest_in)
                    for p_ner in  pred_ner:

                                p_ner= p_ner.data.tolist()

                                y_pred_ner.append(p_ner.index(max(p_ner)))

            f1_ner = f1_score(self.y_true_ner,y_pred_ner, average='micro', labels=self.ner_labels)

            print("epoch:",epoch)
            print("F1 (ner):",f1_ner)
            if f1_ner > best_f1:
                torch.save(self.model.state_dict(), "model_oct_word_2018best1_wordlvl_JUSTner.p")#save best model only
                best_f1 = f1_ner


if __name__ == '__main__':
    best2010  = Best2010NECTEC()
    trainer = TrainModel(LSTMTagger, best2010)
    trainer.train(40)
