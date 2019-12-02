import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CTBTagger(nn.Module):

    def __init__(self,char_embedding_dim, word_embedding_dim, hidden_dim, char_size, word_size, tagset_size_pos, tagset_size_pos2):
        super(CTBTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)

        self.word_embeddings = nn.Embedding(word_size, word_embedding_dim)

        self.lstm_char = nn.LSTM(char_embedding_dim, hidden_dim // 2, bidirectional = True)
        #pos tag
        #word_dim+hidden dim of lstm_char
        self.lstm_pos_pos2 = nn.LSTM(word_embedding_dim+hidden_dim, hidden_dim // 2, bidirectional = True)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag_pos = nn.Linear(hidden_dim, tagset_size_pos)
        self.hidden2tag_pos2 = nn.Linear(hidden_dim, tagset_size_pos2)




    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device),
                torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device))

    def init_hidden_pos_pos2(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device),
                torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device))


    def _lstm_compose(self, char_seq):
        """char composition and concat to word emb"""
        self.hidden = self.init_hidden()
        char_embeds = self.char_embeddings(char_seq)
        lstm_out, self.hidden = self.lstm_char(
        char_embeds.view(len(char_seq), 1, -1), self.hidden)
        return  lstm_out

    def _lstm_pos_pos2_feats(self,word_char_cat):
        self.hidden_pos_pos2 = self.init_hidden_pos_pos2()

        lstm_out, self.hidden_pos_pos2 = self.lstm_pos_pos2(word_char_cat.view(len(word_char_cat), 1, -1), self.hidden_pos_pos2)

        tag_space_pos = self.hidden2tag_pos(lstm_out.view(len(word_char_cat), -1))
        tag_space_pos2 = self.hidden2tag_pos2(lstm_out.view(len(word_char_cat), -1))
        return tag_space_pos, tag_space_pos2


    def _max_only_output(self,input_tensor):
        max_one_hot = torch.ones(input_tensor.size())
        _,i=torch.max(input_tensor, 1)
        result = torch.zeros(input_tensor.size()).scatter_(1,i.cpu(),max_one_hot) #scatter does not work on GPU ???
        return result.to(device)


    def forward(self, text_word_seq,text_char_seq):
        word_emb=self.word_embeddings(text_word_seq)
        char_tensor_list=[]
        for word_char_seq in text_char_seq:
            lstm_out = self._lstm_compose(word_char_seq)
            #concat last fwd output with last backward output
            lstm_out_fwd = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_dim // 2)[-1,:,0] #view(seq_len, batch, num_directions, hidden_size)
            lstm_out_bwd = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], 2, self.hidden_dim // 2)[0,:,1]
            char_tensor_list.append(torch.cat((lstm_out_fwd, lstm_out_bwd),1))

        # if len(char_tensor_list)<1:
        #     print(char_tensor_list,text_word_seq,text_char_seq)
        char_tensor_list = torch.cat(tuple(char_tensor_list))

        #crate backoff vector here (word+char)
        word_char_cat = torch.cat((word_emb,char_tensor_list ), 1)

        lstm_pos_feats, lstm_pos2_feats = self._lstm_pos_pos2_feats(word_char_cat)

        #tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=self.tagset_size_pos)
        tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=1)
        tag_scores_pos2= F.log_softmax(lstm_pos2_feats,dim=1)

        return tag_scores_pos,  tag_scores_pos2
    #    return lstm_char_feats, embedded, lstm_out # TESTING
