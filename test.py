import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
# digits = load_digits()
cover_type_dataset = fetch_covtype()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cover_type_dataset.data, cover_type_dataset.target, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model architecture, warm_start are to use model.fit sequentially, max_iter=1 is in order to run only one epoch for each model.fit
model_with_adam = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', alpha=0.001, max_iter=1, batch_size=128, learning_rate_init=0.001, shuffle=False, verbose=True, warm_start=True, random_state=1, n_iter_no_change=1000)
model_with_aadam = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='custom_aadam', alpha=0.001, max_iter=1, batch_size=128, learning_rate_init=0.001, shuffle=False, verbose=True, warm_start=True, random_state=1, n_iter_no_change=1000)
model_with_adamw = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='custom_adamw', alpha=0.001, max_iter=1, batch_size=128, learning_rate_init=0.001, shuffle=False, verbose=True, warm_start=True, random_state=1, n_iter_no_change=1000)


# Train the model and save the best model with the highest test set accuracy
# best_model = None
best_adam_test_acc = 0
best_adam_epoch_num = 1
best_aadam_test_acc = 0
best_aadam_epoch_num = 1
best_adamw_test_acc = 0
best_adamw_epoch_num = 1
test_adam_accs = []
test_aadam_accs = []
test_adamw_accs = []

"""
Limit train test size:
with no limit one epoch will be enough for over 95% accuracy, and we couldn't compare our solver behaviour efficiently
Now with train size of 100 and batch size of 10 we have 10 (=100/10) calls to solver each epoch
"""
TRAIN_SET_SIZE = 1000
NUM_EPOCHS = 30
for i in range(1,NUM_EPOCHS+1):
    model_with_adam.fit(X_train[:TRAIN_SET_SIZE], y_train[:TRAIN_SET_SIZE], )
    model_with_aadam.fit(X_train[:TRAIN_SET_SIZE], y_train[:TRAIN_SET_SIZE], )
    model_with_adamw.fit(X_train[:TRAIN_SET_SIZE], y_train[:TRAIN_SET_SIZE], )

    test_acc_adam = model_with_adam.score(X_test, y_test)
    test_acc_aadam = model_with_aadam.score(X_test, y_test)
    test_acc_adamw = model_with_adamw.score(X_test, y_test)

    test_adam_accs.append(test_acc_adam)
    test_aadam_accs.append(test_acc_aadam)
    test_adamw_accs.append(test_acc_adamw)

    if test_acc_adam > best_adam_test_acc:
        best_adam_test_acc = test_acc_adam
        best_adam_epoch_num = i
    if test_acc_aadam > best_aadam_test_acc:
        best_aadam_test_acc = test_acc_aadam
        best_aadam_epoch_num = i
    if test_acc_adamw > best_adamw_test_acc:
        best_adamw_test_acc = test_acc_adamw
        best_adamw_epoch_num = i


        # best_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam', alpha=0.001, max_iter=1, batch_size=128, learning_rate_init=0.001)
        # best_model.coefs_ = model.coefs_
        # best_model.intercepts_ = model.intercepts_
    print(f'Epoch: {i} Adam Test accuracy: {test_acc_adam}')
    print(f'Epoch: {i} AAdam Test accuracy: {test_acc_aadam}')
    print(f'Epoch: {i} AdamW Test accuracy: {test_acc_adamw}')

print(f'Best Adam test set accuracy: {best_adam_test_acc} found at epoch #{best_adam_epoch_num}')
print(f'Best AAdam test set accuracy: {best_aadam_test_acc} found at epoch #{best_aadam_epoch_num}')
print(f'Best AdamW test set accuracy: {best_adamw_test_acc} found at epoch #{best_adamw_epoch_num}')


# Create a new figure and add a subplot
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot the first score array with a blue line
ax.plot(test_adam_accs, color='blue', linestyle='-', linewidth=2, label='Adam')

# Plot the second score array with a red line
ax.plot(test_aadam_accs, color='red', linestyle='--', linewidth=2, label='AAdam')

# plot AdamW
ax.plot(test_adamw_accs, color='green', linestyle='solid', linewidth=2, label='AdamW')

# Add a title, x-label, and y-label to the plot
ax.set_title('Model Scores')
ax.set_xlabel('Epochs')
ax.set_xlim(1, NUM_EPOCHS)
ax.set_ylabel('Accuracy')

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()
fig.savefig('myplot.png', dpi=300)

# # Plot the test set accuracy after each epoch
# plt.plot(test_adam_accs)
# plt.xlabel('Epoch')
# plt.ylabel('Test set accuracy')
# plt.show()
