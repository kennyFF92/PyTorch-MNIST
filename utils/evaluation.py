import torch
import numpy as np


def multiclass_conf_matrix(preds, labels, classes):
    conf_matrix_list = []
    for c in range(classes):
        pred_pos = len(preds[preds == c])
        pred_neg = len(preds) - pred_pos
        true_pos = len(preds[(preds == c) & (labels == c)])
        false_pos = pred_pos - true_pos
        true_neg = len(preds[(preds != c) & (labels != c)])
        false_neg = pred_neg - true_neg
        conf_matrix_list.append([true_pos, false_pos, true_neg, false_neg])

    return np.array(conf_matrix_list, dtype=np.int32)


def f1_score(conf_matrix):
    f1 = 0.0
    for c, matrix in enumerate(conf_matrix):
        tp, fp, _, fn = matrix
        if tp == 0:
            class_f1 = 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            class_f1 = 2 * prec * rec / (prec + rec)
        f1 += class_f1
        # print("Class {} F1 score: {:.3f}".format(c, class_f1))

    return f1 / len(conf_matrix)


def eval_networks(net_list,
                  classes,
                  batch_size,
                  test_loader,
                  torch_device=torch.device('cpu')):
    ens_logits = np.zeros((len(test_loader.dataset), classes))
    for net_counter, net in enumerate(net_list):
        net.eval()
        test_acc = 0.0
        test_conf = np.zeros((classes, 4), dtype=np.int32)
        batch_counter = 0
        for test_images, test_labels in test_loader:
            test_images = test_images.to(torch_device)
            test_labels = test_labels.to(torch_device)

            test_logits, _ = net(test_images)
            if len(test_images) == batch_size:
                ens_logits[batch_size * batch_counter:batch_size *
                           (batch_counter +
                            1)] += test_logits.cpu().detach().numpy()
            else:
                ens_logits[-len(test_images):] += test_logits.cpu().detach(
                ).numpy()

            test_preds = torch.argmax(test_logits, 1)
            correct_preds = test_preds.eq(test_labels)
            test_acc += torch.sum(correct_preds).item()
            test_conf += multiclass_conf_matrix(test_preds, test_labels,
                                                classes)
            batch_counter += 1

        test_acc /= len(test_loader.dataset)
        test_f1 = f1_score(test_conf)
        print("Network {} - Accuracy: {:.3f} | F1: {:.3f}".format(
            net_counter + 1, test_acc, test_f1))

    ens_acc = 0.0
    ens_conf = np.zeros((classes, 4), dtype=np.int32)
    ens_preds = np.argmax(ens_logits, 1)
    batch_counter = 0
    for _, test_labels in test_loader:
        test_labels = test_labels.numpy()
        if len(test_labels) == batch_size:
            ens_preds_batch = ens_preds[batch_size * batch_counter:batch_size *
                                        (batch_counter + 1)]
        else:
            ens_preds_batch = ens_preds[-len(test_labels):]
        ens_correct_preds = np.equal(ens_preds_batch, test_labels)
        ens_acc += np.sum(ens_correct_preds)
        ens_conf += multiclass_conf_matrix(ens_preds_batch, test_labels,
                                           classes)
        batch_counter += 1

    ens_acc /= len(test_loader.dataset)
    ens_f1 = f1_score(ens_conf)
    print("Ensemble - Accuracy: {:.3f} | F1: {:.3f}".format(ens_acc, ens_f1))
