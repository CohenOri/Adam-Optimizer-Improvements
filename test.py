import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

digits = load_digits()  # Load the MNIST dataset (Easy dataset)
cover_type_dataset = fetch_covtype()  # Load the Cover Type dataset (Harder dataset)


def split_dataset_to_train_test(dataset_data, dataset_target):
    X_train, X_test, y_train, y_test = train_test_split(dataset_data, dataset_target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Split the data into training and testing sets
cover_X_train, cover_X_test, cover_y_train, cover_y_test = split_dataset_to_train_test(cover_type_dataset.data,
                                                                                       cover_type_dataset.target)
digits_X_train, digits_X_test, digits_y_train, digits_y_test = split_dataset_to_train_test(digits.images, digits.target)


def preprocess_cover_dataset(cover_X_train, cover_X_test):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(cover_X_train), scaler.transform(cover_X_test)


def preprocess_digits_dataset(digits_X_train, digits_X_test):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    # Scale the pixel values to be between 0 and 1
    return scaler.fit_transform(digits_X_train.reshape(-1, 64)), scaler.transform(digits_X_test.reshape(-1, 64))


cover_X_train, cover_X_test = preprocess_cover_dataset(cover_X_train, cover_X_test)
digits_X_train, digits_X_test = preprocess_digits_dataset(digits_X_train, digits_X_test)


def construct_model(solver: str, architecture: str = 'regular'):
    """
    :return:
    Multi layer perceptron classifier with 3 layers, with given solver,
    defined to run one iteration (==Epoch in sklearn) each time .fit() is called.
    with Warm Start set to true = remembering the result from previous .fit() call
    - this way we can call .fit() multiple times, each time is an epoch and test the accuracy after each epoch.
    """
    if architecture == 'regular':
        layers = (512, 256, 128)  # 3 layers
    elif architecture == 'deep':
        layers = (256, 128, 128, 64, 64, 32)  # 6 layers
    elif architecture == 'ultra-deep':
        layers = (1024, 512, 256, 128, 64, 64, 32, 32, 16)  # 9 layers
    else:
        raise ValueError(f'No such architecture: {architecture}')

    model = MLPClassifier(hidden_layer_sizes=layers,
                          activation='relu',
                          solver=solver,
                          alpha=0.001, max_iter=1, batch_size=128, learning_rate_init=0.001, shuffle=False,
                          verbose=True,
                          warm_start=True,
                          random_state=1,
                          n_iter_no_change=1000)
    return model


def init_params(architecture: str):
    # Construct models
    model_with_adam = construct_model('adam', architecture)
    model_with_aadam = construct_model('custom_aadam', architecture)
    model_with_adamw = construct_model('custom_adamw', architecture)
    model_with_sgd = construct_model('sgd', architecture)

    # Initialize parameters
    models_list = [model_with_adam, model_with_aadam, model_with_adamw, model_with_sgd]
    models_test_accuracy = [[], [], [], []]  # empty list for each model
    models_best_test_accuracy = [{"best_acc": 0}, {"best_acc": 0}, {"best_acc": 0}, {"best_acc": 0}]
    models_best_epoch_num = [{"best_epoch": 1}, {"best_epoch": 1}, {"best_epoch": 1}, {"best_epoch": 1}]
    models_names = ["Adam", "Aadam", "AdamW", "SGD"]

    return models_list, models_test_accuracy, models_best_test_accuracy, models_best_epoch_num, models_names


models_list, models_test_accuracy, models_best_test_accuracy, models_best_epoch_num, models_names = init_params('deep')


def train_models(models_list: list,
                 models_test_accuracy: list,
                 models_best_test_accuracy: list,
                 models_best_epoch_num: list,
                 models_names: list,
                 dataset_X_train, dataset_y_train,
                 dataset_X_test, dataset_y_test,
                 TRAIN_SET_SIZE: int = 1000, NUM_EPOCHS: int = 30):
    """
    Limit train test size:
    To speed up learn speed (or in MNIST digits one epoch will be enough for over 95% accuracy and we couldn't compare our solver behaviour efficiently)
    Note: with train size of 100 and batch size of 10 we have 10 (=100/10) calls to solver each epoch
    """

    for model, model_acc_on_test, model_best_test_accuracy, model_best_epoch_num, model_name in \
            zip(models_list, models_test_accuracy, models_best_test_accuracy, models_best_epoch_num, models_names):

        for i in range(1, NUM_EPOCHS + 1):
            model.fit(dataset_X_train[:TRAIN_SET_SIZE], dataset_y_train[:TRAIN_SET_SIZE], )
            TEST_SET_SIZE = round(0.2 * TRAIN_SET_SIZE)
            model_test_acc = model.score(dataset_X_test[:TEST_SET_SIZE], dataset_y_test[:TEST_SET_SIZE])
            model_acc_on_test.append(model_test_acc)

            if model_test_acc > model_best_test_accuracy["best_acc"]:
                model_best_test_accuracy["best_acc"] = model_test_acc
                model_best_epoch_num["best_epoch"] = i

            print(f'Model {model_name} Epoch: #{i} Test accuracy: {model_test_acc}')

    for model, model_acc_on_test, model_best_test_accuracy, model_best_epoch_num, model_name \
            in zip(models_list, models_test_accuracy, models_best_test_accuracy, models_best_epoch_num, models_names):
        print(f'Best {model_name} test set accuracy: {model_best_test_accuracy} found at epoch #{model_best_epoch_num}')


NUM_EPOCHS = 40
TRAIN_SET_SIZE = 100000

train_models(models_list, models_test_accuracy, models_best_test_accuracy, models_best_epoch_num, models_names,
             cover_X_train, cover_y_train, cover_X_test, cover_y_test, TRAIN_SET_SIZE=TRAIN_SET_SIZE, NUM_EPOCHS=NUM_EPOCHS)

dataset_test_adam_accs = models_test_accuracy[0]
dataset_test_aadam_accs = models_test_accuracy[1]
dataset_test_adamw_accs = models_test_accuracy[2]
dataset_test_sgd_accs = models_test_accuracy[3]


def plot_model_scores(test_adam_accs,
                      test_aadam_accs,
                      test_adamw_accs,
                      test_sgd_accs,
                      dataset_name: str,
                      NUM_EPOCHS_FOR_SCLAE: int = 30):
    # Create a new figure and add a subplot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot Adam
    ax.plot(test_adam_accs, color='blue', linestyle='solid', linewidth=2, label='Adam')
    # Plot AAdam
    ax.plot(test_aadam_accs, color='red', linestyle='--', linewidth=2, label='AAdam')
    # plot AdamW
    ax.plot(test_adamw_accs, color='green', linestyle='--', linewidth=2, label='AdamW')
    # plot SGD
    ax.plot(test_sgd_accs, color='yellow', linestyle='solid', linewidth=2, label='SGD')

    # Add a title, x-label, and y-label to the plot
    ax.set_title(f'Model Scores over {dataset_name}')
    ax.set_xlabel('Epochs')
    ax.set_xlim(1, NUM_EPOCHS_FOR_SCLAE)
    ax.set_ylabel('Accuracy')
    # Add a legend to the plot
    ax.legend()
    # Display the plot
    plt.show()
    fig.savefig(f'datasets_results_plots/{dataset_name}_scores.png', dpi=300)


plot_model_scores(dataset_test_adam_accs, dataset_test_aadam_accs, dataset_test_adamw_accs, dataset_test_sgd_accs,
                  dataset_name="Cover_Type", NUM_EPOCHS_FOR_SCLAE=NUM_EPOCHS)
