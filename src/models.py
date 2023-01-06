import torch
import torch.nn as nn
from src.config import Config
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from src.multihead_attention import MultiheadAttention


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        super(MULTModel, self).__init__()
        self.config = Config()
        self.config.print_config()
        self.LSTM_A = torch.nn.LSTM(input_size=self.config.text_input, hidden_size=self.config.LSTM_hidden)
        self.LSTM_B = torch.nn.LSTM(input_size=self.config.audio_input, hidden_size=self.config.LSTM_hidden)
        self.LSTM_C = torch.nn.LSTM(input_size=self.config.video_input, hidden_size=self.config.LSTM_hidden)
        self.dropout = nn.Dropout(self.config.LSTM_dropout)
        self.linear_w1 = nn.Linear(self.config.LSTM_hidden, self.config.text_out_size)
        self.linear_w2 = nn.Linear(self.config.text_out_size, self.config.text_out_size)
        self.relu = nn.ReLU()
        self.DNN = nn.Sequential(
            nn.Dropout(p=self.config.DNNdrop),
            nn.Linear((self.config.text_out_size + 1) *
                      (self.config.text_out_size + 1) *
                      (self.config.text_out_size + 1)
                      , self.config.fusionLine),
            nn.ReLU()
        )
        self.DNN_Fusion = nn.Sequential(
            nn.Dropout(p=self.config.fusionDrop),
            nn.Linear(self.config.fusionLine *
                      self.config.fusionLine
                      , self.config.fusionLine),
            nn.ReLU(),
            nn.Linear(self.config.fusionLine, self.config.fusionLine),
            nn.ReLU(),
            nn.Linear(self.config.fusionLine, self.config.squeezeLineOutput),
        )

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.config.LSTM_hidden, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, t, a, v):
        w_output, (w_hidden, _) = self.LSTM_A(t.transpose(0, 1))
        a_output, (a_hidden, _) = self.LSTM_B(a.transpose(0, 1))
        v_output, (v_hidden, _) = self.LSTM_C(v.transpose(0, 1))
        w_output_ = w_output.permute(1, 0, 2)
        a_output_ = a_output.permute(1, 0, 2)
        v_output_ = v_output.permute(1, 0, 2)
        attn_output_aw = self.attention_net(w_output_, w_hidden)
        attn_output_vw = self.attention_net(v_output_, v_hidden)
        attn_output_wa = self.attention_net(w_output_, a_hidden)
        attn_output_va = self.attention_net(v_output_, a_hidden)
        attn_output_wv = self.attention_net(w_output_, v_hidden)
        attn_output_av = self.attention_net(a_output_, v_hidden)
        aw_hidden = self.dropout(attn_output_aw)
        vw_hidden = self.dropout(attn_output_vw)
        wa_hidden = self.dropout(attn_output_wa)
        va_hidden = self.dropout(attn_output_va)
        wv_hidden = self.dropout(attn_output_wv)
        av_hidden = self.dropout(attn_output_av)
        aw = self.linear_w1(aw_hidden)
        vw = self.linear_w1(vw_hidden)
        wa = self.linear_w1(wa_hidden)
        va = self.linear_w1(va_hidden)
        wv = self.linear_w1(wv_hidden)
        av = self.linear_w1(av_hidden)
        aw = self.relu(aw)
        vw = self.relu(vw)
        wa = self.relu(wa)
        va = self.relu(va)
        wv = self.relu(wv)
        av = self.relu(av)
        aw = self.linear_w2(aw)
        vw = self.linear_w2(vw)
        wa = self.linear_w2(wa)
        va = self.linear_w2(va)
        wv = self.linear_w2(wv)
        av = self.linear_w2(av)
        aw = self.relu(aw)
        vw = self.relu(vw)
        wa = self.relu(wa)
        va = self.relu(va)
        wv = self.relu(wv)
        av = self.relu(av)
        batch_size = a.shape[0]
        # 改：下一行原本是DTYPE = torch.cuda.FloatTensor
        DTYPE = torch.FloatTensor
        aw = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), aw), dim=1)
        vw = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), vw), dim=1)
        wa = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), wa), dim=1)
        va = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), va), dim=1)
        wv = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), wv), dim=1)
        av = torch.cat((torch.ones((batch_size, 1), requires_grad=False).type(DTYPE), av), dim=1)
        fusion_tensor_1 = torch.bmm(aw.unsqueeze(2), vw.unsqueeze(1))
        fusion_tensor_1 = fusion_tensor_1.view(-1, (aw.shape[1]) * (vw.shape[1]), 1)
        fusion_tensor_1 = torch.bmm(fusion_tensor_1, wa.unsqueeze(1)).view(batch_size, -1)
        fusion_tensor_2 = torch.bmm(va.unsqueeze(2), wv.unsqueeze(1))
        fusion_tensor_2 = fusion_tensor_2.view(-1, (va.shape[1]) * (wv.shape[1]), 1)
        fusion_tensor_2 = torch.bmm(fusion_tensor_2, av.unsqueeze(1)).view(batch_size, -1)
        fusion_tensor_1 = self.DNN(fusion_tensor_1)
        fusion_tensor_2 = self.DNN(fusion_tensor_2)
        fusion_tensor = torch.bmm(fusion_tensor_1.unsqueeze(2), fusion_tensor_2.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1)
        ans = torch.sigmoid(self.DNN_Fusion(fusion_tensor))
        self.output_range = torch.FloatTensor([6]).type(DTYPE)
        self.output_shift = torch.FloatTensor([-3]).type(DTYPE)
        ans = ans * self.output_range + self.output_shift
        return ans


if __name__ == '__main__':
    torch.cuda.empty_cache()
    hyp_params = 1
    config = Config()
    # 下一行改：原本是model = MULTModel(hyp_params).to(config.device)
    model = MULTModel(hyp_params)
    # 后面 3行改：删掉末尾的.to(config.device)
    data_t = torch.randn([config.batch_size, config.align_length, config.text_input])
    data_a = torch.randn([config.batch_size, config.align_length, config.audio_input])
    data_v = torch.randn([config.batch_size, config.align_length, config.video_input])
    ans = model(data_t, data_a, data_v)
    print("")
    print("*" * 50)
    print(ans.shape)