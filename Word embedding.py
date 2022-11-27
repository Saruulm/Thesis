import numpy as np
import os
os.getcwd()

with open(r"/Users/saruul/Desktop/Thesis/Word embedding/dataset/reviews.txt") as f:
    raw_reviews = f.readlines()
with open(r'/Users/saruul/Desktop/Thesis/Word embedding/dataset/labels.txt') as f:
    raw_labels = f.readlines()


# #### Creating an input vector

tokens = []
for review in raw_reviews:
    review = set(review.split(' '))
    review.remove('')
    tokens.append(list(review))
len(tokens[1])

words = set()
for review in tokens:
    for word in review:
        words.add(word)
words = list(words)


len(words)

word_to_index = {}
for i, word in enumerate(words):
    word_to_index[word] = i
len(word_to_index)

input_dataset = np.zeros((len(tokens), len(words)))

for i, review in enumerate(tokens):
    for word in review:
        input_dataset[i, word_to_index[word]] = 1

word_to_index['of']

input_dataset[1][66451]
input_dataset[0]

target_dataset = np.array([])
for label in raw_labels:
    if label == 'positive\n':
        target_dataset = np.append(target_dataset, 1)
    else:
        target_dataset = np.append(target_dataset, 0)
target_dataset.shape

target_dataset = target_dataset.reshape(25000, 1)

train_dataset = input_dataset[:24000]
train_labels = target_dataset[:24000]

test_dataset = input_dataset[24000:]
test_labels = target_dataset[24000:]
train_dataset.shape


# ### Network

# #### Linear Layer
class Layer_Linear:
    """Representing a neural network layer"""
    
    def __init__(self, n_inputs, n_outputs):
        """Initlize weights and bias"""
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))
    
    def forward(self, inputs):
        """
        It multiplies the inputs by the weights 
        and then sums them, and then sums bias.
        """
        #To calculate gradient, remembering input values
        self.inputs = inputs
        #Calculate outputs' values
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        """Gradient with respect to parameters and input"""
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dresults = np.dot(dvalues, self.weights.T)


# #### Activation functions
class Activation_ReLU:
    """ReLU activation"""
    
    def forward(self, inputs):
        """Forward pass"""
        
        #To calculate gradient, remembering input values
        self.inputs = inputs
        
        #Calculate outputs' values
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        """Backward pass"""
        
        self.dresults = self.inputs > 0
        self.dresults = self.dresults * dvalues



class Activation_Sigmoid:
    """Sigmoid activation"""
    
    def forward(self, inputs):
        """Forward pass"""
        
        #Calculate outputs' values
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        """Backward pass"""
        
        self.dresults = dvalues * (1 - self.output) * self.output


# #### Loss function

class Loss_MSE():
    """MSE Loss function"""
    
    def forward(self, y_pred, y_true):
        """Forward pass"""     
        error = np.mean((y_pred - y_true) ** 2)
        return error
    
    def backward(self, y_pred, y_true):
        """Derivative of MSE with respect to preds"""
        
        #Number of samples
        samples = len(y_pred)
        
        #Number of output nodes
        outputs = len(y_pred[0])
        
        #Derivative of MSE
        self.dresults = 2 * (y_pred - y_true) / (outputs * samples)


# #### Optimizer

class Optimizer_GD:
    """Gradient descent optimizer"""
    
    def __init__(self, alpha=1.):
        """Initialize hyperparameters"""
        self.alpha = alpha

    def update_parameters(self, layer):
        """Update parameters"""
        
        weights_delta = layer.dweights * self.alpha
        biases_delta = layer.dbiases * self.alpha
        
        #Update parameters
        layer.weights -= weights_delta
        layer.biases -= biases_delta


# ### Hyperparameter
max_epoch = 5
alpha = 1
batch_size = 128


# ### Initialize the model

layer1 = Layer_Linear(len(words), 100)
activation1 = Activation_ReLU()

layer2 = Layer_Linear(100, 1)
activation2 = Activation_Sigmoid()


# #### Initlize optimizer and loss function

loss = Loss_MSE()
optimizer = Optimizer_GD(alpha)


# ### Training the model
train_steps = len(train_dataset) // batch_size
if train_steps * batch_size < len(train_dataset):
    train_steps += 1


t1 = datetime.now()

for epoch in range(max_epoch):
    train_error = 0
    train_accuracy = 0
    
    for i in range(train_steps):
        batch_start = i * batch_size
        batch_end = (i+1) * batch_size
        
        input = train_dataset[batch_start:batch_end]
        true = train_labels[batch_start:batch_end]
        
        #Forward pass
        layer1.forward(input)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        train_error += loss.forward(activation2.output, true) / train_steps
        train_accuracy += np.mean((np.abs(activation2.output - true) < 0.5)) / train_steps
        
        #Backward pass
        loss.backward(activation2.output, true)
        activation2.backward(loss.dresults)
        layer2.backward(activation2.dresults)
        activation1.backward(layer2.dresults)
        layer1.backward(activation1.dresults)
        
        #Update parameters
        optimizer.update_parameters(layer2)
        optimizer.update_parameters(layer1)

    print(f'epoch: {epoch},',
          f'Train error: {train_error:.3f},',
          f'Train accuracy: {train_accuracy:.3f}')
    
t2 = datetime.now()
print(t2-t1)


# #### Testing the model

test_steps = len(test_dataset) // batch_size
if test_steps * batch_size < len(test_dataset):
    test_steps += 1


test_error = 0
test_accuracy = 0

for i in range(test_steps):
    batch_start = i * batch_size
    batch_end = (i+1) * batch_size
    
    input = test_dataset[batch_start:batch_end]
    true = test_labels[batch_start:batch_end]
    
    layer1.forward(input)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    test_error += loss.forward(activation2.output, true) / test_steps
    test_accuracy += np.mean((np.abs(activation2.output - true) < 0.5)) / test_steps


print(f'Test error: {test_error:.3f},',
      f'Test accuracy: {test_accuracy:.3f}')


# ## 2. Hidden layers arrange the inputs into n groups

from collections import Counter

def similar(target):
    target_index = word_to_index[target]
    scores = Counter()
    for word, index in word_to_index.items():
        # Finding Euclidian distance
        scores[word] = -np.linalg.norm(layer1.weights[index] - layer1.weights[target_index])
    
    return scores.most_common(10)


similar('beautiful')


word_to_index['awful']

layer1.weights[49117]



