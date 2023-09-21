import model
import utils
import torch
import pickle
import matplotlib.pyplot as plt
import os

labels = ['English', 'Irish', 'Chinese', 'French', 'Portuguese', 'Greek', 'German', 'Dutch', 'Russian', 'Vietnamese', 'Arabic', 'Spanish', 'Scottish', 'Japanese', 'Italian', 'Korean', 'Polish', 'Czech']

# f = open('store.pckl','rb')
# category_lines,category_labels, labels = pickle.load(f)
# f.close()

# Loading the data
category_lines,category_labels, labels = utils.loadata()

# # Creating the model
# model = model1.RNN(utils.N_LETTERS, 128, len(labels))

model = model.RNN(utils.N_LETTERS, 128, 18)
# model.load_state_dict(torch.load("model_state"))

# Fit the data in model
model.fit((category_lines,category_labels, labels), epochs=3000, lr = 0.001)

# Saving the model state
torch.save(model.state_dict(), "model_state")


model.eval()

f = open('history.pckl','rb')
train_loss_history, test_loss_history = pickle.load(f)
f.close()

# print(model.predict("Young",labels))

plt.subplot(1,2,1)
plt.plot(train_loss_history)
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.plot(test_loss_history)
plt.title("Test Loss")

plt.show()



# # Sample Predict 
# print(model.predict('Young',labels))

