import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from basil.select_device import device

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','self']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
        elif self.method == 'self':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
            self.w = torch.nn.Parameter(torch.FloatTensor(hidden_size))


    def self_score(self, encoder_output):
        energy = self.attn(encoder_output).tanh()

        return torch.sum(self.w * energy, dim=2)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'self':
            attn_energies = self.self_score( encoder_outputs)


        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LSTMTagger(nn.Module):

    def __init__(self,char_embedding_dim, syl_embedding_dim , word_embedding_dim, hidden_dim, char_size, syl_size, word_size, prefix_size, tagset_size_pos, tagset_size_ner):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        self.syl_embeddings = nn.Embedding(syl_size, syl_embedding_dim)

        self.word_embeddings = nn.Embedding(word_size, word_embedding_dim)

        self.prefix_embeddings = nn.Embedding(prefix_size, hidden_dim)

        self.attn_char = Attn("self",self.hidden_dim)
        self.attn_syl = Attn("self",self.hidden_dim)


        self.lstm_char = nn.LSTM(char_embedding_dim, hidden_dim // 2, bidirectional = True)

        self.lstm_syl = nn.LSTM(syl_embedding_dim, hidden_dim // 2, bidirectional = True)
        #pos tag
        #word_dim+hidden dim of lstm_char
        self.lstm_pos_ner = nn.LSTM(word_embedding_dim+hidden_dim*2, hidden_dim // 2, bidirectional = True)


        # The linear layer that maps from hidden state space to tag space

        self.hidden2tag_pos = nn.Linear(hidden_dim , tagset_size_pos)
        self.hidden2tag_ner = nn.Linear(hidden_dim , tagset_size_ner)



    def init_hidden(self,feat):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #(h0,c0)
        return (self.prefix_embeddings(feat).view(2,1,self.hidden_dim//2).expand(2,1,self.hidden_dim//2),
                torch.zeros(2, 1, self.hidden_dim //2,requires_grad=True).to(device))



    def init_hidden_pos_ner(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device),
                torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device))


    def _lstm_compose_char(self, char_seq, feat):
        """char composition and concat to word emb"""

        #hidden = (self.prefix_embeddings(feat).view(1,1,self.hidden_dim),torch.zeros(1, 1, self.hidden_dim,requires_grad=True))
        hidden = self.init_hidden(feat)
        char_embeds = self.char_embeddings(char_seq)
        lstm_out, _ = self.lstm_char(
        char_embeds.view(len(char_seq), 1, -1), hidden)
        attn_weight=self.attn_char(None,lstm_out)
        context = attn_weight.bmm(lstm_out.transpose(0, 1))
        return  context

    def _lstm_compose_syl(self, syl_seq, feat):
        """syllable composition and concat to word emb"""

        #hidden = (self.prefix_embeddings(feat).view(1,1,self.hidden_dim),torch.zeros(1, 1, self.hidden_dim,requires_grad=True))
        hidden = self.init_hidden(feat)
        syl_embeds = self.syl_embeddings(syl_seq)
        lstm_out, _ = self.lstm_syl(
        syl_embeds.view(len(syl_seq), 1, -1), hidden)
        attn_weight=self.attn_syl(None,lstm_out)
        context = attn_weight.bmm(lstm_out.transpose(0, 1))
        return  context


    def _lstm_pos_ner_feats(self,word_char_cat,prefix_emb):
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

    def forward(self, text_word_seq, text_syl_seq, text_char_seq, text_feat_seq):
        word_emb=self.word_embeddings(text_word_seq)
        prefix_emb=self.prefix_embeddings(text_feat_seq)
        char_tensor_list=[]
        for word_char_seq,word_feat in zip(text_char_seq,text_feat_seq):
            lstm_out = self._lstm_compose_char(word_char_seq,word_feat)
            #concat last fwd output with last backward output
            # lstm_out_fwd = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_dim // 2)[-1,:,0] #view(seq_len, batch, num_directions, hidden_size)
            # lstm_out_bwd = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_dim // 2)[0,:,1]
            char_tensor_list.append(lstm_out)

        #SYLLABLE STUFF
        syl_tensor_list=[]
        for word_syl_seq,word_feat in zip(text_syl_seq,text_feat_seq):
            syl_lstm_out = self._lstm_compose_syl(word_syl_seq, word_feat)
            #concat last fwd output with last backward output
            # syl_lstm_out_fwd = syl_lstm_out.view(syl_lstm_out.shape[0], syl_lstm_out.shape[1], 2, self.hidden_dim // 2)[-1,:,0] #view(seq_len, batch, num_directions, hidden_size)
            # syl_lstm_out_bwd = syl_lstm_out.view(syl_lstm_out.shape[0], syl_lstm_out.shape[1], 2, self.hidden_dim // 2)[0,:,1]
            syl_tensor_list.append(syl_lstm_out)

        char_tensor_list = torch.cat(tuple(char_tensor_list))
        syl_tensor_list = torch.cat(tuple(syl_tensor_list))

        #crate backoff vector here (word+char)
        word_char_cat = torch.cat((word_emb,char_tensor_list.squeeze(1),syl_tensor_list.squeeze(1)), 1)

        lstm_pos_feats, lstm_ner_feats = self._lstm_pos_ner_feats(word_char_cat,prefix_emb)

        #tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=self.tagset_size_pos)
        tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=1)
        tag_scores_ner= F.log_softmax(lstm_ner_feats,dim=1)

        return tag_scores_pos,  tag_scores_ner
    #    return lstm_char_feats, embedded, lstm_out # TESTING
