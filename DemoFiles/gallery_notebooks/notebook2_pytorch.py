PATH = "C:/Users/adoko/data/"

import pandas as pd
import ast

def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except:
        return []
def replace_nulls_in_list(lst):
    return [0.0 if x is None else x for x in lst]

if __name__ == '__main__':
    testPDF = pd.read_parquet(path= PATH + 'testUndersampled.snappy.parquet',
                              columns=['HLF_input', 'encoded_label'])

    trainPDF = pd.read_parquet(path= PATH + 'trainUndersampled.snappy.parquet',
                               columns=['HLF_input', 'encoded_label'])

    ##Preprocessing train

    trainPDF['HLF_input'] = trainPDF['HLF_input'].apply(string_to_list)
    trainPDF['encoded_label'] = trainPDF['encoded_label'].apply(string_to_list)

    trainPDF['HLF_input'] = trainPDF['HLF_input'].apply(replace_nulls_in_list)
    trainPDF['encoded_label'] = trainPDF['encoded_label'].apply(replace_nulls_in_list)

    trainPDF = trainPDF[['HLF_input', 'encoded_label']]

    ##Preprocessing test
    testPDF['HLF_input'] = testPDF['HLF_input'].apply(string_to_list)
    testPDF['encoded_label'] = testPDF['encoded_label'].apply(string_to_list)

    testPDF['HLF_input'] = testPDF['HLF_input'].apply(replace_nulls_in_list)
    testPDF['encoded_label'] = testPDF['encoded_label'].apply(replace_nulls_in_list)

    testPDF = testPDF[['HLF_input', 'encoded_label']]

    # Check the number of events in the train and test datasets

    num_test = testPDF.count()
    num_train = trainPDF.count()

    print('There are {} events in the test dataset'.format(num_test))
    print('There are {} events in the train dataset'.format(num_train))


    import numpy as np

    X = np.stack(trainPDF["HLF_input"])
    y = np.stack(trainPDF["encoded_label"])

    X_test = np.stack(testPDF["HLF_input"])
    y_test = np.stack(testPDF["encoded_label"])


    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import TensorDataset, DataLoader

    torch.__version__

    torch.cuda.is_available()


    class Net(nn.Module):
        def __init__(self, nh_1, nh_2, nh_3):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(14, nh_1)
            self.fc2 = nn.Linear(nh_1, nh_2)
            self.fc3 = nn.Linear(nh_2, nh_3)
            self.fc4 = nn.Linear(nh_3, 3)

        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            output = nn.functional.softmax(self.fc4(x), dim=1)
            return output

    def create_model(nh_1, nh_2, nh_3):
        model = Net(nh_1, nh_2, nh_3)
        return model

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                target = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_accuracy))
        return(test_loss, test_accuracy)


    def train(model, device, train_loader, optimizer, epoch):
        log_interval = 10000
        model.train()
        correct = 0
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            # metrics
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            #
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader.dataset), loss.item()))

        # train_loss = loss.item()
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100. * correct / len(train_loader.dataset)
        print('\nTrain set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader.dataset), train_accuracy))

        return(train_loss, train_accuracy)


    torch.manual_seed(1)

    # device = torch.device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = {'batch_size': 128}
    test_kwargs = {'batch_size': 1000}
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    # Map train and test data to Pytorch's dataloader
    train_tensor = TensorDataset(torch.Tensor(X),torch.Tensor(y))
    test_tensor =  TensorDataset(torch.Tensor(X_test),torch.Tensor(y_test))
    train_loader = torch.utils.data.DataLoader(train_tensor, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_tensor, **test_kwargs)

    model = create_model(50,20,10).to(device)
    optimizer = optim.Adam(model.parameters())


    def train_loop():
        gamma = 0.7
        epochs = 5
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        hist = {}
        hist['loss'] = []
        hist['accuracy'] = []
        hist['val_loss'] = []
        hist['val_accuracy'] = []
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
            val_loss, val_accuracy = test(model, device, test_loader)
            scheduler.step()
            hist['loss'] += [train_loss]
            hist['accuracy'] += [train_accuracy]
            hist['val_loss'] += [val_loss]
            hist['val_accuracy'] += [val_accuracy]
        return(hist)

    hist = train_loop()


    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    # Graph with loss vs. epoch

    plt.figure()
    plt.plot(hist['loss'], label='train')
    plt.plot(hist['val_loss'], label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.title("HLF classifier loss")
    plt.show()
