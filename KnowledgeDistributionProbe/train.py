import gc
import os
import pickle
import warnings
import json

import math
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm

import config
from dataset import ClassifierDataset
from model import LlamaMLPClassifier
from utils import args_util, log_util
from utils.log_util import logger

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 12})
rng = np.random.default_rng(42)

def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy

def create_Xs_and_ys(datasets, scores, val_test_splits=None, test_only=False, no_val=False):
    # Data splitting for sklearn linear models
    if val_test_splits is None:
        val_test_splits = [0.2, 0.1]
    X = np.array(datasets)
    y = np.array(scores)

    if test_only:
        X_tests, y_tests = [], []

        for i in range(X.shape[0]):
            X_tests.append(X[i])
            y_tests.append(y)
        return None, None, X_tests, None, None, y_tests

    valid_size = val_test_splits[0]
    test_size = val_test_splits[1]

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []

    for i in range(X.shape[0]):
        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X[i], y, test_size=test_size, random_state=42)
        X_tests.append(X_test)
        y_tests.append(y_test)
        if no_val:
            X_trains.append(X_train_val)
            y_trains.append(y_train_val)
            continue
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size,
                                                          random_state=42)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_vals.append(X_val)
        y_vals.append(y_val)

    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests


def bootstrap_func(y_true, y_score, func):
    y_tuple = (y_true, y_score)

    metric_i = func(*y_tuple)
    metric_dict = {}
    metric_dict['mean'] = metric_i
    metric_dict['bootstrap'] = compatible_bootstrap(
        func, rng)(*y_tuple)  # a bit slow to run

    return metric_dict


def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data,), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }

    return inner


def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)

    return converted_func


# Train and evaluation function.
def sklearn_train_and_evaluate(model, X_train, y_train, X_valid, y_valid, silent=False):
    model.fit(X_train, y_train)

    # Calculate training loss and score
    train_probs = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_probs)

    # Calculate validation loss
    valid_preds = model.predict(X_valid)
    valid_probs = model.predict_proba(X_valid)
    valid_loss = log_loss(y_valid, valid_probs)
    val_accuracy = np.mean((valid_preds == y_valid).astype(int))
    auroc_score = roc_auc_score(y_valid, valid_probs[:, 1])
    if not silent:
        print(f"Validation Accuracy: {val_accuracy:.4f}, AUROC: {auroc_score:.4f}")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")


def sklearn_evaluate_on_test(model, X_test, y_test, silent=False, bootstrap=True):
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_loss = log_loss(y_test, test_probs)
    test_accuracy = np.mean((test_preds == y_test).astype(int))
    test_f1 = f1_score(y_test, test_preds, average="weighted")

    if bootstrap:
        auroc_score = bootstrap_func(y_test, test_probs[:, 1], auroc)
        auroc_score_scalar = auroc_score['mean']
    else:
        auroc_score = auroc_score_scalar = roc_auc_score(y_test, test_probs[:, 1])

    if not silent:
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, AUROC: {auroc_score_scalar:.4f}")

    return test_loss, test_accuracy, auroc_score, test_f1


def train_single_metric_llama_mlp(D, token_type='tbg', metric='b_entropy'):
    """train and test on single metric (e.g. SE, Acc) on single dataset"""
    var_name = token_type[0] + metric[0]
    hidden_size = D.tbg_dataset[0].size(-1)

    hidden_states = getattr(D, f'{token_type}_dataset')
    labels = getattr(D, metric)

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_Xs_and_ys(
        hidden_states, labels
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    accs = []
    aucs = []
    f1s = []

    # Llama MLP
    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(
            zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
        if i == 0:
            continue

        train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)
        model_save_path = os.path.join(config.SAVE_PATH, 'results', 'models',
                                       f'llama_mlp_{i}.pt')
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        if os.path.exists(model_save_path):
            print(f"Loading model from {model_save_path}")
            model = torch.load(model_save_path)
            model.to(device).eval()
        else:
            print(f"Training on {D.name}-{token_type.upper()}-{metric.upper()} {i + 1}/{len(X_trains)}")
            model = LlamaMLPClassifier(input_dim=hidden_size).to(device)
            model = llama_mlp_train_and_evaluate(model, train_loader, val_loader, device)

        test_acc, test_f1, test_auc = llama_mlp_test_model(model, test_loader, device)
        accs.append(test_acc)
        aucs.append(test_auc)
        f1s.append(test_f1)

        # Save model
        torch.save(model.cpu(), model_save_path)

        del train_loader, val_loader, test_loader, model
        gc.collect()

    result = Dataset()
    setattr(result, f'{var_name}_accs', accs)
    setattr(result, f'{var_name}_aucs', aucs)
    setattr(result, f'{var_name}_f1s', f1s)

    return result


def train_single_metric_logistic(D, token_type='tbg', metric='b_entropy'):
    """train and test on single metric (e.g. SE, Acc) on single dataset"""
    var_name = token_type[0] + metric[0]

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_Xs_and_ys(
        getattr(D, f'{token_type}_dataset'), getattr(D, metric)
    )

    accs = []
    aucs = []
    f1s = []
    models = []

    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(
            zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
        if i == 0:
            continue
        print(f"Training on {D.name}-{token_type.upper()}-{metric.upper()} {i + 1}/{len(X_trains)}")
        model = LogisticRegression()
        sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
        test_loss, test_acc, test_auc, test_f1 = sklearn_evaluate_on_test(model, X_test, y_test)
        accs.append(test_acc)
        aucs.append(test_auc)
        f1s.append(test_f1)
        models.append(model)

    result = Dataset()
    setattr(result, f'{var_name}_accs', accs)
    setattr(result, f'{var_name}_aucs', aucs)
    setattr(result, f'{var_name}_f1s', f1s)
    setattr(result, f'{var_name}_models', models)

    return result


# simple get-around for unpacking bootstrapping dicts
auc = lambda aucs: [ac['mean'] for ac in aucs]
idf = lambda x: x  # identical function


# Best split for SE binarization.
def best_split(entropy: torch.Tensor, label="Dx"):
    """
    Identify best split for minimizing reconstruction error via low and high SE mean estimates,
    as discussed in Section 4. Binarization of paper (ArXiv: 2406.15927)
    """
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split

        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])

        mse = np.sum((ents[low_idxs] - low_mean) ** 2) + np.sum((ents[high_idxs] - high_mean) ** 2)
        mse = np.sum(mse)

        split_mses.append(mse)

    split_mses = np.array(split_mses)

    plt.plot(splits, split_mses, label=label)
    return splits[np.argmin(split_mses)]


def binarize_entropy(entropy, thres=0.0):  # 0.0 means even splits for normalized entropy scores
    """Binarize entropy scores into 0s and 1s"""
    binary_entropy = torch.full_like(entropy, -1, dtype=torch.float)
    binary_entropy[entropy < thres] = 0
    binary_entropy[entropy > thres] = 1

    return binary_entropy


def trinarize_entropy(entropy, thres1=0.639, thres2=1.478):
    """Trinarize entropy scores into 0's and 1's"""
    trinary_entropy = torch.full_like(entropy, -1, dtype=torch.float)
    trinary_entropy[entropy < thres1] = 0
    trinary_entropy[entropy > thres2] = 1
    return trinary_entropy


def llama_mlp_train_and_evaluate(model, train_loader, val_loader, device, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-5, weight_decay=0.0)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss, train_correct = 0.0, 0
        total_train_samples = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train_samples += inputs.size(0)

        train_loss /= total_train_samples
        train_accuracy = train_correct / total_train_samples

        val_loss, val_accuracy = llama_mlp_evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return model


def llama_mlp_evaluate(model, data_loader, criterion, device):
    model.eval()
    val_loss, val_correct = 0.0, 0
    total_val_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            total_val_samples += inputs.size(0)

    val_loss /= total_val_samples
    val_accuracy = val_correct / total_val_samples
    return val_loss, val_accuracy


def llama_mlp_test_model(model, test_loader, device, bootstrap=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_probs_pos_class = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_probs_pos_class.append(probs[:, 1].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_probs_pos_class = np.concatenate(all_probs_pos_class)

    y_test = all_labels
    test_preds = all_preds
    test_probs = all_probs
    test_probs_positive = all_probs_pos_class

    test_loss = log_loss(y_test, test_probs)

    test_accuracy = accuracy_score(y_test, test_preds)

    test_f1 = f1_score(y_test, test_preds, average="weighted")

    if bootstrap:

        auroc_score = bootstrap_func(y_test, test_probs_positive, auroc)
        auroc_score_scalar = auroc_score['mean']
    else:
        auroc_score = auroc_score_scalar = roc_auc_score(y_test, test_probs_positive)

    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, AUROC: {auroc_score_scalar:.4f}")
    return test_accuracy, test_f1, auroc_score


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(load_train_dataset_cache=True):
    load_train_dataset_cache = load_train_dataset_cache and os.path.exists(
        os.path.join(config.SAVE_PATH, 'cache', 'train-dataset.pkl'))
    if load_train_dataset_cache:
        with open(os.path.join(config.SAVE_PATH, 'cache', 'train-dataset.pkl'), 'rb') as f:
            D = pickle.load(f)
        logger.info('Loaded cached train dataset')
    else:
        with open(os.path.join(config.SAVE_PATH, 'results', 'full-result.pkl'), 'rb') as f:
            entropy_result_dict = pickle.load(f)
        with open(os.path.join(config.SAVE_PATH, 'results', 'hidden-states.pkl'), 'rb') as f:
            hidden_states_dict = pickle.load(f)
        D = ClassifierDataset(entropy_result_dict, hidden_states_dict)
        with open(os.path.join(config.SAVE_PATH, 'cache', 'train-dataset.pkl'), 'wb') as f:
            pickle.dump(D, f)
        logger.info('Loaded entropy results and created cached train dataset')

    all_entropy = D.entropy
    split = best_split(all_entropy, "All datasets collective")
    plt.legend()
    plt.title('Sum of squared errors at different splits')
    D.b_entropy = binarize_entropy(D.entropy, split)
    D.split = split
    print(f"Dummy accuracy for {D.name}: {max(torch.mean(D.b_entropy).item(), 1 - torch.mean(D.b_entropy).item()):.4f}")

    unsure_question_ids = [D.question_ids[i] for i in range(len(D.b_entropy)) if D.b_entropy[i] >= 1]
    sure_question_ids = [D.question_ids[i] for i in range(len(D.b_entropy)) if D.b_entropy[i] <= 0]
    with open(os.path.join(config.SAVE_PATH, 'results', 'split_question_ids.json'), 'w') as f:
        json.dump({'unsure': unsure_question_ids, 'sure': sure_question_ids}, f)

    full_result = dict()

    baseline_result = dict()
    validation_is_false = [1 - i for i in D.accuracies]

    liks = torch.tensor([np.mean(record) for record in D.log_likelihood])
    accs = torch.tensor(D.accuracies)
    probs = np.exp(liks)
    processed_accs = []
    processed_probs = []
    for acc, prob in zip(accs, probs):
        if math.isnan(acc) or math.isnan(prob):
            logger.warn("Nan value in log-likelihood or accuracy, acc:{} prob:{}".format(acc, prob))
            continue
        processed_accs.append(acc)
        processed_probs.append(prob)
    processed_accs = torch.tensor(processed_accs)
    loglik_tuple = (processed_accs, processed_probs)
    baseline_result['baseline_log_likelihood'] = auroc(*loglik_tuple)

    regular_entropy_list = []
    for item in D.full_answers:
        log_liks = [np.mean(record['logprobs']) for record in item]
        regular_entropy_list.append(predictive_entropy(log_liks))
    regular_entropy_tuple = (validation_is_false, regular_entropy_list)
    baseline_result['baseline_regular_entropy'] = auroc(*regular_entropy_tuple)

    p_false_fixed_tuple = (validation_is_false, D.p_false_fixed)
    baseline_result['baseline_p_true'] = auroc(*p_false_fixed_tuple)

    semantic_entropy_tuple = (validation_is_false, D.entropy)
    baseline_result['baseline_semantic_entropy'] = auroc(*semantic_entropy_tuple)

    full_result['baseline_result'] = baseline_result

    with open(os.path.join(config.SAVE_PATH, 'results', 'baseline-result.pkl'), 'wb') as f:
        pickle.dump(baseline_result, f)
        print('Saved baseline results')

    print("Baseline results:")
    print(f"Baseline log-likelihood: {baseline_result['baseline_log_likelihood']}")
    print(f"Baseline regular entropy: {baseline_result['baseline_regular_entropy']}")
    print(f"Baseline p_true: {baseline_result['baseline_p_true']}")
    print(f"Baseline semantic entropy: {baseline_result['baseline_semantic_entropy']}")

    # sep logistic
    if not os.path.exists(os.path.join(config.SAVE_PATH, 'results', 'train-result-logistic.pkl')):
        train_result_logistic = train_single_metric_logistic(D, 'tbg', 'b_entropy')
        full_result['train_result_logistic'] = train_result_logistic
        with open(os.path.join(config.SAVE_PATH, 'results', 'train-result-logistic.pkl'), 'wb') as f:
            pickle.dump(train_result_logistic, f)
            print('Saved logistic regression results')
    else:
        with open(os.path.join(config.SAVE_PATH, 'results', 'train-result-logistic.pkl'), 'rb') as f:
            train_result_logistic = pickle.load(f)
            full_result['train_result_logistic'] = train_result_logistic

    # Ours
    if not os.path.exists(os.path.join(config.SAVE_PATH, 'results', 'train-result-llama-mlp.pkl')):
        train_result_llama_mlp = train_single_metric_llama_mlp(D, 'tbg', 'b_entropy')
        full_result['train_result_llama_mlp'] = train_result_llama_mlp
        with open(os.path.join(config.SAVE_PATH, 'results', 'train-result-llama-mlp.pkl'), 'wb') as f:
            pickle.dump(train_result_llama_mlp, f)
            print('Saved llama mlp results')
    else:
        with open(os.path.join(config.SAVE_PATH, 'results', 'train-result-llama-mlp.pkl'), 'rb') as f:
            train_result_llama_mlp = pickle.load(f)
            full_result['train_result_llama_mlp'] = train_result_llama_mlp

    # accuracy probes logistic
    if not os.path.exists(os.path.join(config.SAVE_PATH, 'results', 'train-result-accuracy-logistic.pkl')):
        train_result_logistic = train_single_metric_logistic(D, 'tbg', 'accuracies')
        full_result['train_result_accuracy_logistic'] = train_result_logistic
        with open(os.path.join(config.SAVE_PATH, 'results', 'train-result-accuracy-logistic.pkl'), 'wb') as f:
            pickle.dump(train_result_logistic, f)
            print('Saved accuracy logistic regression results')
    else:
        with open(os.path.join(config.SAVE_PATH, 'results', 'train-result-accuracy-logistic.pkl'), 'rb') as f:
            train_result_logistic = pickle.load(f)
            full_result['train_result_accuracy_logistic'] = train_result_logistic

    with open(os.path.join(config.SAVE_PATH, 'results', 'train-full-result.pkl'), 'wb') as f:
        pickle.dump(full_result, f)
        print('Saved full results')


if __name__ == "__main__":
    parser = args_util.get_parser()
    args, unknown = parser.parse_known_args()
    if args.config is not None:
        config.update_config(args.config, continue_training=True)

    log_util.init_logger()
    logger.info('Starting new run with args: %s', args)
    train()
