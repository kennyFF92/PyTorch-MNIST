import os
import cv2
import torch
import numpy as np
import albumentations as A

from torch.utils.tensorboard import SummaryWriter
from .evaluation import f1_score, multiclass_conf_matrix
from .params import get_start_time


def augment_function(p=0.5):
    c = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.3,
                           rotate_limit=40,
                           border_mode=cv2.BORDER_REPLICATE,
                           p=0.8)
    ],
                  p=p)
    return c


def train_network(net,
                  cost,
                  optimizer,
                  classes,
                  batch_size,
                  train_loader,
                  val_loader=None,
                  epochs=30,
                  lr_sched=None,
                  torch_device=torch.device('cpu'),
                  tb_path='./runs',
                  save_model=True,
                  save_path='./models',
                  ensemble_session_id=0):
    # Tensorboard init
    start_time = get_start_time()
    tensorboard_path = os.path.join(tb_path, ensemble_session_id)
    os.makedirs(tensorboard_path, exist_ok=True)
    tb_writer = SummaryWriter(os.path.join(tensorboard_path, start_time))
    train_iter = val_iter = 0
    augment = augment_function(0.9)

    # Training
    for epoch in range(epochs):
        net.train()
        train_loss = train_acc = 0.0
        for train_images, train_labels in train_loader:
            # Augmentation
            train_images_aug = np.array(
                [augment(image=img.numpy())['image'] for img in train_images])
            train_images = torch.from_numpy(train_images_aug).float().to(
                torch_device)
            train_labels = train_labels.to(torch_device)

            # Forward + Loss
            optimizer.zero_grad()
            train_logits, _ = net(train_images)
            loss = cost(train_logits, train_labels)
            train_preds = torch.argmax(train_logits, 1)
            correct_preds = train_preds.eq(train_labels)

            # Backward + Optimization
            loss.backward()
            optimizer.step()

            # Update stats loss
            batch_loss = loss.item()
            train_loss += batch_loss
            train_acc += torch.sum(correct_preds).item()

            batch_loss /= batch_size
            train_iter += 1
            if train_iter % 200 == 0:
                tb_writer.add_scalar('BatchLoss', batch_loss, train_iter)
                print("Current batch loss: {:.5f}".format(batch_loss))

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        tb_writer.add_scalar('TrainingLoss', train_loss, train_iter)
        tb_writer.add_scalar('TrainingAccuracy', train_acc, train_iter)
        summary_string = "--- Epoch {}:\n\tTraining loss: {:.5f}".format(
            epoch + 1, train_loss)
        summary_string += " | Training accuracy {:.3f}".format(train_acc)

        if lr_sched is not None:
            lr_sched.step()
            tb_writer.add_scalar('LearningRate',
                                 lr_sched.get_lr()[0], train_iter)

        # Validation
        if val_loader is not None:
            net.eval()
            val_loss = val_acc = 0.0
            val_conf = np.zeros((classes, 4), dtype=np.int32)
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(
                    torch_device), val_labels.to(torch_device)
                val_logits, _ = net(val_images)
                val_preds = torch.argmax(val_logits, 1)
                loss = cost(val_logits, val_labels)
                correct_preds = val_preds.eq(val_labels)

                val_loss += loss.item()
                val_acc += torch.sum(correct_preds).item()
                val_conf += multiclass_conf_matrix(val_preds, val_labels,
                                                   classes)
                val_iter += 1

            val_loss /= len(val_loader.dataset)
            val_acc /= len(val_loader.dataset)
            val_f1 = f1_score(val_conf)
            tb_writer.add_scalar('ValidationLoss', val_loss, val_iter)
            tb_writer.add_scalar('ValidationAccuracy', val_acc, val_iter)
            tb_writer.add_scalar('ValidationF1', val_f1, val_iter)
            summary_string += "\n\tValidation loss: {:.5f}".format(val_loss)
            summary_string += " | Validation accuracy {:.3f}".format(val_acc)
            summary_string += " | Validation F1 {:.3f}".format(val_f1)
            print(summary_string)

    tb_writer.add_graph(net, train_images)
    tb_writer.close()

    if save_model:
        trained_model_path = os.path.join(save_path, ensemble_session_id)
        os.makedirs(trained_model_path, exist_ok=True)
        torch.save(net.state_dict(),
                   os.path.join(trained_model_path, start_time + '.pth'))
