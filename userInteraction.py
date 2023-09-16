import torch
import gpt as g

def restore(model):
    model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model


model = g.GPTLanguageModel()
model = restore(model)
# g.train(model)
model.output()
# userInput = input()
# model.output(userInput)