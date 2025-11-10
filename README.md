# nn
Minimal implementation of multilayer perceptrons and backpropagation for the course "**Advanced Concepts in Machine Learning**" (2025) in M.Sc. Artificial Intelligence at Maastricht University

install dependencies 

```
pip install -r requirements.txt
```

Create and train a model, and make predictions 

```
import nn.model
import nn.layer 
import nn.activation
import nn.loss

model = nn.model.Model([
    nn.layer.Layer(256, 128, nn.activation.sigmoid),
    nn.layer.Layer(128, 256, nn.activation.sigmoid)
])

# create or load training data: x_train, y_train

model.fit(x_train, y_train, n_epochs=1_000, batch_size=32, lr=0.01, loss_fn=nn.loss.mse) 

y_pred = model(x_train)
```

# Features

- Modular neural network architecture
- Multiple activation functions (Sigmoid, Linear)
- Various loss functions:
    - Mean Squared Error (MSE)
    - Cross Entropy
    - Cross Entropy from Logits
- Gradient checking 
- Mini batch training
- L2 regularization support
