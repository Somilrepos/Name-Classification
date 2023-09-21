import torch
import torch.nn as nn
import sklearn.model_selection
import utils
import pickle
import time

class RNN(nn.Module):
    
    def __init__(self, inputsize, hiddensize, outputsize):
        super(RNN, self).__init__()

        """ intitializer function for RNN 
        @param inputsize: Size of the input 
        @param hiddesize: Size of the hidden layer 
        @param outputsize: Size of the output (aka number of output classes)

        Model Architechture:
            It's a vanilla RNN.
        
            ht = sigmoid[ W*(ht-1 + x)]
            y^t = softmax(W*ht)
        
        """

        self.hiddensize = hiddensize
        self.outputsize = outputsize


        self.h = nn.Linear(inputsize+hiddensize, hiddensize)
        nn.init.xavier_uniform_(self.h.weight, gain=1)

        self.y = nn.Linear(hiddensize, outputsize)
        nn.init.xavier_uniform_(self.y.weight, gain=1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim = 1)


    def forward(self, input, hidden):
        # Forward funtion for RNN

        combined = torch.cat((input,hidden), 1)

        hidden = self.sigmoid(self.h(combined))
        logits = self.y(hidden)
        output = self.softmax(logits)

        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1,self.hiddensize)
        

    def fit(self, train_data, lossfn = nn.NLLLoss(), optimizerfn = torch.optim.Adam, lr = 0.001 ,RANDOM_STATE = 42, batch_size = 32, epochs = 1000, TRAIN_SIZE = 0.8, TEST_SIZE = 0.2):
        """ Training function for RNN.
        Inputs:

        @param train_data*: training data for model.
        @param lossfn(function): lossfn for model. (default: NLLLoss)
        @param optimizer(function): lossfn for model. (default: SGD)
        @param lr(float): learing rate for optimizer. (default: 0.001)
        @param RANDOM_STATE(int): random state for controlling the randomness of data shuffling. (Default: None)
        @param batch_size(int): batch size of each epoch. (default: 32)
        @param epoch(int): Number of epoch to train. (default: 1000)
        @param TRAIN_SIZE(float): a value c, 0<c<1, training dataset split size. (default: 0.8)
        @param Test_SIZE(float): a value c, 0<c<1, testing dataset split size. (default: 0.2)

        Outputs:

        @param train_loss_history: stores history of training losses.
        @param train_loss_history: stores history of testing losses.

        """
        
        # Monitoring
        train_loss_history = []
        test_loss_history = []

        #Initialising optimizer function.
        optimizer = optimizerfn(self.parameters(), lr = lr)

        def accuracy(predictions, true_label):
            score = 0
            total = 0
            # print(torch.argmax(predictions[20]))
            # print(true_label[20])
            for i in range(len(predictions)):
                if true_label[i] == torch.argmax(predictions[i]):
                    score += 1
                total += 1
            return score*100/total

    
        # Training and testing data preparation
        category_lines,category_labels, labels = train_data

        # Converting category_lines in a list of tensors
        X = [utils.line_to_tensor(line) for line in category_lines]

        # Converting category_labels into a tensor of size (D,1)
        y = torch.tensor([labels.index(label) for label in category_labels])
        y = y.unsqueeze(dim=1)

        # Splitting the dataset into Train and Test Data
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE, train_size = TRAIN_SIZE, random_state=RANDOM_STATE)

        curidx = 0
        print("Training Started...")

        # Measuring time
        st = time.time()
        epoch = 0
        while epoch < epochs:

            final_train_loss = 0

            for idx in range(batch_size):
                line_tensor = X_train[curidx+idx]
                prev_line_hidden = self.init_hidden()
                
                for i in range(line_tensor.shape[0]):   
                    train_output, prev_line_hidden = self.forward(line_tensor[i], prev_line_hidden)

                train_loss = lossfn(train_output, y_train[curidx+idx])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                final_train_loss += train_loss.item()
        
            final_train_loss = final_train_loss/batch_size
            
            curidx = (curidx+batch_size)%len(X_train)
            if curidx+batch_size > len(X_train):
                curidx = 0
            
            train_loss_history.append(final_train_loss)

            with torch.no_grad():   
                test_output = torch.tensor([])
                final_test_loss = 0
                for idx in range(len(X_test)):
                    line_tensor = X_test[idx]
                    prev_hidden = self.init_hidden()
                    for i in range(line_tensor.shape[0]):
                        test_line_output, prev_hidden = self.forward(line_tensor[i], prev_hidden)
                    test_loss = lossfn(test_line_output, y_test[idx])
                    final_test_loss += test_loss
                    test_output = torch.cat((test_output,test_line_output), 0)
                final_test_loss = final_test_loss/len(X_test)
                test_loss_history.append(final_test_loss)

                print(f"Epoch: {epoch+1}, train_loss: {final_train_loss}, test_loss: {final_test_loss} Acuracy: {accuracy(test_output, y_test.squeeze(dim = 1))}")
                f = open('history.pckl', 'wb')
                pickle.dump((train_loss_history,test_loss_history), f)
                f.close()
            epoch += 1

        et = time.time()
        print(f"Training Finished... Total Execution Time:{(et-st)} seconds")
        return (train_loss_history, test_loss_history) 
    
    def predict(self, x, labels):
        prev_hidden = self.init_hidden()
        tensor = utils.line_to_tensor(x)
        for i in range(tensor.shape[0]):   
            output, prev_hidden = self.forward(tensor[i], prev_hidden)
        
        return labels[(torch.argmax(output).item())]



            
