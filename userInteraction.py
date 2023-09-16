import gpt as g

model = g.model

userInput = input('Please begin chatting: ')
model.output(userInput)