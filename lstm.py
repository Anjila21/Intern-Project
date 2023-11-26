class ExtractLSTM(nn.Module):
    def init(self, vocab_size, embed_size, hidden_size):
        super().init()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 5)

    def forward(self, inpt):
        embedded = self.embed(inpt)
        feature, _ = self.lstm(embedded)
        oupt = self.linear(feature)
        return oupt