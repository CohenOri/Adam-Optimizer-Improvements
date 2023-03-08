import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
digits = load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=0.2, random_state=42)

# Scale the pixel values to be between 0 and 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 64))
X_test = scaler.transform(X_test.reshape(-1, 64))

# Define the model architecture, warm_start are to use model.fit sequentially, max_iter=1 is in order to run only one epoch for each model.fit
model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', alpha=0.001, max_iter=1, batch_size=10, learning_rate_init=0.001, shuffle=True, verbose=True, warm_start=True, random_state=1, n_iter_no_change=1000)


# Train the model and save the best model with the highest test set accuracy
# best_model = None
best_test_acc = 0
best_epoch_num = 1
test_accs = []

"""
Limit train test size:
with no limit one epoch will be enough for over 95% accuracy, and we couldn't compare our solver behaviour efficiently
"""
TRAIN_SET_SIZE = 100
for i in range(30):
    model.fit(X_train[:TRAIN_SET_SIZE], y_train[:TRAIN_SET_SIZE], )
    test_acc = model.score(X_test, y_test)
    test_accs.append(test_acc)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch_num = i
        # best_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', alpha=0.001, max_iter=1, batch_size=128, learning_rate_init=0.001)
        # best_model.coefs_ = model.coefs_
        # best_model.intercepts_ = model.intercepts_
    print('Epoch', i+1, 'Test accuracy:', test_acc)

print(f'Best test set accuracy: {best_test_acc} found at epoch #{best_epoch_num}')

# Plot the test set accuracy after each epoch
plt.plot(test_accs)
plt.xlabel('Epoch')
plt.ylabel('Test set accuracy')
plt.show()
