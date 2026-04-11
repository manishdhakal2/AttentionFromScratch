
import numpy as np

class Attention:
    def __init__(self):


        pass

    def dot_attention(self,h_enc,h_dec_t):

        """
        Returns the dot product attention of the given encoder and decoder hidden states

        Params :
            h_enc : Encoder hidden state across all timesteps
            h_dec_t :  Current Decoder hidden state  
        """

        similarity_score = np.dot(h_enc, h_dec_t)

        softmaxed_score = self.softmax(similarity_score)

        context_vector = np.dot(h_enc, softmaxed_score)



    def softmax(self,x):

        """
        Returns the softmax of the given input vector
        """
        return np.exp(x)/np.sum(np.exp(x))


