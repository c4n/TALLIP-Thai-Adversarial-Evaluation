import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from allennlp.modules.elmo import Elmo, batch_to_ids

class ORCHIDTagger(nn.Module):

    def __init__(self, word_embedding_dim, hidden_dim, word_size, tagset_size_pos, tagset_size_type):
        super(ORCHIDTagger, self).__init__()

        self.hidden_dim = hidden_dim

        #self.char_embeddings = nn.Embedding(char_size, char_embedding_dim)
        options_file = '/home/can/Documents/thai_nlp_research/BASIL/orchid_options.json'
        weight_file = '/home/can/Documents/thai_nlp_research/BASIL/orchid_weights.hdf5'
       
        #self.word_embeddings = nn.Embedding(word_size, word_embedding_dim)
        self.word_embeddings = Elmo(options_file, weight_file, 2,requires_grad=True, dropout=0.1)
        #self.word_embeddings = Elmo(options_file, weight_file, 2, dropout=0.1)
        #self.lstm_char = nn.LSTM(char_embedding_dim, hidden_dim // 2, bidirectional = True)
        #pos tag
        #word_dim+hidden dim of lstm_char
        self.lstm_pos_type = nn.LSTM(128, hidden_dim // 2, bidirectional = True)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag_pos = nn.Linear(hidden_dim, tagset_size_pos)
        self.hidden2tag_type = nn.Linear(hidden_dim, tagset_size_type)




    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device),
                torch.zeros(2, 1, self.hidden_dim // 2,requires_grad=True).to(device))

    def init_hidden_pos(self):
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

    def _lstm_pos_feats(self,word_char_cat):
        self.hidden_pos = self.init_hidden_pos()


        lstm_out, self.hidden_pos = self.lstm_pos_type(word_char_cat.squeeze().view(len(word_char_cat.squeeze()), 1, -1), self.hidden_pos)


        tag_space_pos = self.hidden2tag_pos(lstm_out.view(len(word_char_cat.squeeze()), -1))
        tag_space_type = self.hidden2tag_type(lstm_out.view(len(word_char_cat.squeeze()), -1))

        return tag_space_pos, tag_space_type



    def _max_only_output(self,input_tensor):
        max_one_hot = torch.ones(input_tensor.size())
        _,i=torch.max(input_tensor, 1)
        result = torch.zeros(input_tensor.size()).scatter_(1,i.cpu(),max_one_hot) #scatter does not work on GPU ???
        return result.to(device)


    def forward(self, text_word_seq):
        word_emb=self.word_embeddings(text_word_seq)
        # char_tensor_list=[]
        # for word_char_seq in text_char_seq:
        #     lstm_out = self._lstm_compose(word_char_seq)
        #     char_tensor_list.append(lstm_out[-1])
        #
        # char_tensor_list = torch.cat(tuple(char_tensor_list))
        #
        # #crate backoff vector here (word+char)
        # word_char_cat = torch.cat((word_emb,char_tensor_list ), 1)

        lstm_pos_feats, lstm_type_feats = self._lstm_pos_feats(word_emb['elmo_representations'][0])


        #tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=self.tagset_size_pos)
        tag_scores_pos= F.log_softmax(lstm_pos_feats,dim=1)
        tag_scores_type= F.log_softmax(lstm_type_feats,dim=1)

        return tag_scores_pos, tag_scores_type
    #    return lstm_char_feats, embedded, lstm_out # TESTING
