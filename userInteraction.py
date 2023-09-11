import torch
import gpt as g

def restore(model):
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    return model


# encoded = g.encode("this better work")
# print(encoded)

model = g.GPTLanguageModel()
model = restore(model)
# g.train(model)
model.output()
# userInput = input()
# model.output(userInput)