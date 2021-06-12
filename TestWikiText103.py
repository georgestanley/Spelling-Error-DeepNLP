from torchtext.datasets import WikiText103
train, val, test = WikiText103(split=('train', 'valid', 'test'))

print(train)