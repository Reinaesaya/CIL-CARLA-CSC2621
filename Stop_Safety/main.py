import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle

from data import *
from models import *

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc


TRAINFOLDER = "./data/Train_Town1/"
VALIDFOLDER = "./data/Val_Town1/"
TESTFOLDER  = "./data/Test_Town2/"

runResNet = True

train = True



if __name__ == "__main__":

    trainInput, trainTarget = loadData(TRAINFOLDER)
    validInput, validTarget = loadData(VALIDFOLDER)
    testInput,  testTarget  = loadData(TESTFOLDER)
    
    print("Number of samples in {}: {}".format(TRAINFOLDER, trainTarget.shape[0]))
    print("Number of samples in {}: {}".format(VALIDFOLDER, validTarget.shape[0]))
    print("Number of samples in {}: {}".format(TESTFOLDER, testTarget.shape[0]))

    train_histories = {}


    if runResNet:
        print("Running ResNet")
        EPOCHS = 25
        BATCH_SIZE = 32
        LEARNING_RATE = 1e-5
        REG_LAMBDA = 1.0
        DROPOUT = 0.5

        ResNet = ResNet_Model()
        if train:
            loss_history, accuracy_history = ResNet.train(trainInput, validInput, trainTarget, validTarget, \
                                                    reg_lambda=REG_LAMBDA, learning_rate=LEARNING_RATE, \
                                                    dropout=DROPOUT, batch_size=BATCH_SIZE, epochs=EPOCHS)

            max_loss = np.max(sum(loss_history.values(), []))

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.title('ResNet Loss')
            plt.plot(range(1,EPOCHS+1), loss_history['train'], 'r', label='Train set')
            plt.plot(range(1,EPOCHS+1), loss_history['test'] , 'b', label='Valid set')
            plt.axis([1, EPOCHS+1, 0, max_loss])
            plt.legend(loc='best')

            plt.subplot(2, 1, 2)
            plt.title('ResNet Accuracy')
            plt.plot(range(1,EPOCHS+1), accuracy_history['train'], 'r', label='Train set')
            plt.plot(range(1,EPOCHS+1), accuracy_history['test'] , 'b', label='Valid set')
            plt.axis([1, EPOCHS+1, 0, 1])
            plt.legend(loc='best')

            train_histories['ResNet'] = {
                'loss_history': loss_history,
                'accuracy_history': accuracy_history,
                'max_loss': max_loss,
            }

        train_loss, train_acc, train_pred = ResNet.test(trainInput, trainTarget)
        valid_loss, valid_acc, valid_pred = ResNet.test(validInput, validTarget)
        test_loss,  test_acc,  test_pred  = ResNet.test(testInput , testTarget)

        # Receiver Operating Characteristic
        fpr_train, tpr_train, _ = roc_curve(trainTarget, train_pred)
        fpr_valid, tpr_valid, _ = roc_curve(validTarget, valid_pred)
        fpr_test,  tpr_test,  _ = roc_curve(testTarget , test_pred)
        roc_auc_train = auc(fpr_train, tpr_train)
        roc_auc_valid = auc(fpr_valid, tpr_valid)
        roc_auc_test  = auc(fpr_test , tpr_test)

        plt.figure()
        plt.plot(fpr_train, tpr_train, \
            label='Train Set ROC Curve (AUC = {:.2f})'.format(roc_auc_train))
        plt.plot(fpr_valid, tpr_valid, \
            label='Valid Set ROC Curve (AUC = {:.2f})'.format(roc_auc_valid))
        plt.plot(fpr_test , tpr_test, \
            label='Test Set ROC Curve (AUC = {:.2f})'.format(roc_auc_test))
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        # Precision Recall
        aps_train = average_precision_score(trainTarget, train_pred)
        aps_valid = average_precision_score(validTarget, valid_pred)
        aps_test  = average_precision_score(testTarget , test_pred)

        # prec_train, rec_train, _ = precision_recall_curve(trainTarget, train_pred)
        # prec_valid, rec_valid, _ = precision_recall_curve(validTarget, valid_pred)
        # pr_auc_train = auc(prec_train, rec_train)
        # pr_auc_valid = auc(prec_valid, rec_valid)

        # plt.figure()
        # plt.plot(prec_train, rec_train, \
        #     label='Train Set PR Curve (AUC = {:.2f})'.format(pr_auc_train))
        # plt.plot(prec_valid, rec_valid, \
        #     label='Test Set PR Curve (AUC = {:.2f})'.format(pr_auc_valid))
        # plt.plot([0, 1], [0.5, 1.5], lw=2, linestyle='--')
        # plt.xlim([0.0, 1.05])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('Precision')
        # plt.ylabel('Recall')
        # plt.title('Precision-Recall')
        # plt.legend(loc="best")

        
        print("-------------------------------------")
        print("Final results from using ResNet Model")
        print("-------------------------------------")
        print("Train set --> Loss: {}; ROC-AUC: {}; Avg-PrecScore: {}".format(train_loss, roc_auc_train, aps_train))
        print("Valid set --> Loss: {}; ROC-AUC: {}; Avg-PrecScore: {}".format(valid_loss, roc_auc_valid, aps_valid))
        print("Test  set --> Loss: {}; ROC-AUC: {}; Avg-PrecScore: {}".format(test_loss , roc_auc_test , aps_test))
        print("-------------------------------------")


    # if train:
    #     with open("./train_histories.pickle", "wb") as f:
    #         pickle.dump(train_histories, f)
    # else:
    #     with open("./train_histories.pickle", "rb") as f:
    #         train_histories = pickle.load(f)

# ### Loss Comparisons ###

#     plt.figure(figsize=(8, 8))
#     plt.subplot(2, 1, 1)
#     plt.title('Training Loss')
#     max_loss = 0
#     for k in train_histories.keys():
#         loss_history = train_histories[k]['loss_history']
#         max_loss = max(max_loss, train_histories[k]['max_loss'])
#         plt.plot(range(1,EPOCHS+1), loss_history['train'], label=k)
#     plt.axis([1, EPOCHS+1, 0, max_loss])
#     plt.legend(loc='best')

#     plt.subplot(2, 1, 2)
#     plt.title('Test Loss')
#     max_loss = 0
#     for k in train_histories.keys():
#         loss_history = train_histories[k]['loss_history']
#         max_loss = max(max_loss, train_histories[k]['max_loss'])
#         plt.plot(range(1,EPOCHS+1), loss_history['test'], label=k)
#     plt.axis([1, EPOCHS+1, 0, max_loss])
#     plt.legend(loc='best')

# ### Accuracy Comparisons ###

#     plt.figure(figsize=(8, 8))
#     plt.subplot(2, 1, 1)
#     plt.title('Training Accuracy')
#     for k in train_histories.keys():
#         accuracy_history = train_histories[k]['accuracy_history']
#         plt.plot(range(1,EPOCHS+1), accuracy_history['train'], label=k)
#     plt.axis([1, EPOCHS+1, 0, 1])
#     plt.legend(loc='best')

#     plt.subplot(2, 1, 2)
#     plt.title('Test Accuracy')
#     for k in train_histories.keys():
#         accuracy_history = train_histories[k]['accuracy_history']
#         plt.plot(range(1,EPOCHS+1), accuracy_history['test'], label=k)
#     plt.axis([1, EPOCHS+1, 0, 1])
#     plt.legend(loc='best')



    plt.show()
