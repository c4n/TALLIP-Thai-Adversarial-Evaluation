import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from basil.select_device import device

class LSTMTagger(nn.Module):

    def __init__(self,char_embedding_dim, word_embedding_dim, hidden_dim, char_size, word_size, prefix_size, tagset_size_pos, tagset_size_ner):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        self.word_embeddings = nn.Embedding(word_size, word_embedding_dim)

        self.prefix_embeddings = nn.Embedding(prefix_size, hidden_dim)

        self.lstm_char = nn.LSTM(char_embedding_dim, hidden_dim)
        #pos tag
        #word_dim+hidden dim of lstm_char
        self.lstm_pos_ner = nn.LSTM(word_embedding_dim+hidden_dim, hidden_dim // 2, bidirectional = True)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag_pos = nn.Linear(hidden_dim, tagset_size_pos)
        self.hidden2tag_ner = nn.Linear(hidden_dim, tagset_size_ner)




    def init_hidden(self,feat):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #(h0,c0)
        return (self.prefix_embeddings(feat).view(1,1,self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim,requires_grad=True).to(device))


    def init_hidden_pos_ner(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers*num_dir, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device),
                torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device))


    def _lstm_compose(self, char_seq, feat):
        """char composition and concat to word emb"""

        #hidden = (self.prefix_embeddings(feat).view(1,1,self.hidden_dim),torch.zeros(1, 1, self.hidden_dim,requires_grad=True))
        hidden = self.init_hidden(feat)
        char_embeds = self.char_embeddings(char_seq)
        lstm_out, _ = self.lstm_char(
        char_embeds.view(len(char_seq), 1, -1), hidden)
        return  lstm_out

    def _lstm_pos_ner_feats(self,word_char_cat):
        self.hidden_pos_ner = self.init_hidden_pos_ner()

        lstm_out, self.hidden_pos_ner = self.lstm_pos_ner(word_char_cat.view(len(word_char_cat), 1, -1), self.hidden_pos_ner)

        tag_space_pos = self.hidden2tag_pos(lstm_out.view(len(word_char_cat), -1))
        tag_space_ner = self.hidden2tag_ner(lstm_out.view(len(word_char_cat), -1))
        return tag_space_pos, tag_space_ner


    def _max_only_output(self,input_tensor):
        max_one_hot = torch.ones(input_tensor.size())
        _,i=torch.max(input_tensor, 1)
        result = torch.zeros(input_tensor.size()).scatter_(1,i.cpu(),max_one_hot) #scatter does not work on GPU ???
        return result.to(device)

    def forward(self, text_word_seq, text_char_seq, text_feat_seq):
        word_emb=self.word_embeddings(text_word_seq)
        char_tensor_list=[]
        for word_char_seq,word_feat in zip(text_char_seq,text_feat_seq):
            lstm_out = self._lstm_compose(word_char_seq,word_feat)
            char_tensor_list.append(lstm_out[-1])

        char_tensor_list = torch.cat(tuple(char_tensor_list))

        #crate backoff vector here (word+char)
        word_char_cat = torch.cat((word_emb,char_tensor_list ), 1)

        lstm_pos_feats, lstm_ner_feats = self._lstm_pos_ner_feats(word_char_cat)

        #tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=self.tagset_size_pos)
        tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=1)
        tag_scores_ner= F.log_softmax(lstm_ner_feats,dim=1)

        return tag_scores_pos,  tag_scores_ner
    #    return lstm_char_feats, embedded, lstm_out # TESTING
