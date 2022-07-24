# template for attention synthesizers
class DenseAttention(nn.Module):
    def __init__(self, max_seq_len, d_k, d_hid = 64, attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2=64
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(d_k, d_hid) #由原始input的维度d_k到中间的维度d_hid
        self.w_2 = nn.Linear(d_hid, max_seq_len) #由中间的维度d_hid到设置的最大维度N
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None):

        # b x n x lq x dq -> b x n x lq x lq #
        dense_attn = self.w_2(self.relu(self.w_1(q)))[:,:,:,:len_q]
        # print('Attn: ', dense_attn.shape)
        # print('Mask: ', mask.shape)
        # print('V: ', v.shape)

        #这里貌似是只学了q，因为是self-attention，所以学习它都别人的映射，只要学习q或者k当中的一个就够了

        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)

        dense_attn = self.dropout(F.softmax(dense_attn, dim=-1))
        output = torch.matmul(dense_attn, v)
        
        return output, dense_attn

# template for attention synthesizers
class DenseAttentionpre(nn.Module):
    def __init__(self, max_seq_len, d_k, d_hid = 64, attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2=64
        super(DenseAttentionpre, self).__init__()
        self.w_1 = nn.Linear(d_k, d_hid) #由原始input的维度d_k到中间的维度d_hid
        self.w_2 = nn.Linear(d_hid, max_seq_len) #由中间的维度d_hid到设置的最大维度N
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None):

        # b x n x lq x dq -> b x n x lq x lq #
        dense_attn = self.w_2(self.relu(self.w_1(q)))[:,:,:,:len_q]
        # print('Attn: ', dense_attn.shape)
        # print('Mask: ', mask.shape)
        # print('V: ', v.shape)

        #这里貌似是只学了q，因为是self-attention，所以学习它都别人的映射，只要学习q或者k当中的一个就够了

        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask == 0, -1e9)

        dense_attn = self.dropout(F.softmax(dense_attn, dim=-1))
        output = torch.matmul(dense_attn, v)
        
        return output, dense_attn