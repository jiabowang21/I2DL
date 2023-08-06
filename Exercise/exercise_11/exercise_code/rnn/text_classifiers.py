import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        # if you do not inherit from lightning module use the following line
        self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        # self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, 2)
        self.sig = nn.Sigmoid()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        embedded = self.embedding(sequence)
        if lengths is not None:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.to('cpu'))
        else:
            packed_embedded = embedded
        
        outputrnn, (hidden, cell) = self.rnn(packed_embedded)
        output = self.fc(hidden)
        
        batch_size = sequence.size()[1]
        output = output.view(batch_size, -1)
        output = output[:, -1]
        output = self.sig(output)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output
