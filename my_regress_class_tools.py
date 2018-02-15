import torch
from torch.autograd import Variable
import operator
import scipy.io as sio
from new_vet import Hopenet


def train_processing(model, train_loader, num_epochs, criterion, reg_criterion, softmax, idx_tensor, alpha,
                     THRESHHOLD_VAL, optimizer, model_save_name, validation_loader, gpu_id, writer, BATCH_SIZE,
                     len_train_data, len_val_data, scheduler):
    print('Ready to train network.')
    for epoch in range(num_epochs):
        # scheduler.step()
        # ---------------------------testing--------------------------------
        train_acc = 0
        for i, (name, images, labels, cont_labels) in enumerate(train_loader):

            images = Variable(images).cuda()

            # Binned labels
            label_yaw = Variable(labels[:, 0].long()).cuda()
            label_pitch = Variable(labels[:, 1].long()).cuda()
            label_roll = Variable(labels[:, 2].long()).cuda()

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:, 0]).cuda()
            label_pitch_cont = Variable(cont_labels[:, 1]).cuda()
            label_roll_cont = Variable(cont_labels[:, 2]).cuda()

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            # predicted angel
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 2 - 179
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 2 - 179
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 2 - 179

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * 0.1 * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.Tensor(1).cuda() for _ in range(len(loss_seq))]

            for num_image in range(images.shape[0]):
                accuracy_yaw = torch.add(yaw_predicted[num_image], -label_yaw_cont[num_image])
                accuracy_pitch = torch.add(pitch_predicted[num_image], -label_pitch_cont[num_image])
                accuracy_roll = torch.add(roll_predicted[num_image], -label_roll_cont[num_image])

                c1 = operator.gt(THRESHHOLD_VAL, float(torch.abs(accuracy_yaw)))
                c2 = operator.gt(THRESHHOLD_VAL, float(torch.abs(accuracy_pitch)))
                c3 = operator.gt(THRESHHOLD_VAL, float(torch.abs(accuracy_roll)))
                if (c1 & c2 & c3) == True:
                    train_acc += 1
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            n = 100
            if (i + 1) % n == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, Acc: %.6f'
                      % (epoch + 1, num_epochs, i + 1, len_train_data // BATCH_SIZE, loss_yaw.data[0],
                         loss_pitch.data[0], loss_roll.data[0], train_acc / (BATCH_SIZE * n)))
                train_acc = 0

                writer.add_scalar('Train/Loss_yaw', loss_yaw, epoch)
                writer.add_scalar('Train/Loss_pitch', loss_pitch, epoch)
                writer.add_scalar('Train/Loss_roll', loss_roll, epoch)
                writer.add_scalar('Train/Accuracy', train_acc / (BATCH_SIZE * n), epoch)

        # ---------------------------evaluation--------------------------------
        model.eval()
        eval_loss_yaw = 0
        eval_loss_pitch = 0
        eval_loss_roll = 0
        eval_acc = 0.
        for i, (name, images, labels, cont_labels) in enumerate(validation_loader):
            images = Variable(images.cuda(), volatile=True)

            # Binned labels
            label_yaw = Variable((labels[:, 0].long()).cuda(), volatile=True)
            label_pitch = Variable((labels[:, 1].long()).cuda(), volatile=True)
            label_roll = Variable((labels[:, 2].long()).cuda(), volatile=True)

            # Continuous labels
            label_yaw_cont = Variable((cont_labels[:, 0]).cuda(), volatile=True)
            label_pitch_cont = Variable((cont_labels[:, 1]).cuda(), volatile=True)
            label_roll_cont = Variable((cont_labels[:, 2]).cuda(), volatile=True)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            # Exception
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 2 - 180
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 2 - 180
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 2 - 180

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            eval_loss_yaw += loss_yaw
            eval_loss_pitch += loss_pitch
            eval_loss_roll += loss_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.Tensor(1).cuda() for _ in range(len(loss_seq))]

            for num_image in range(images.shape[0]):
                accuracy_yaw = torch.add(yaw_predicted[num_image], -label_yaw_cont[num_image])
                accuracy_pitch = torch.add(pitch_predicted[num_image], -label_pitch_cont[num_image])
                accuracy_roll = torch.add(roll_predicted[num_image], -label_roll_cont[num_image])

                c1 = operator.gt(THRESHHOLD_VAL, float(torch.abs(accuracy_yaw)))
                c2 = operator.gt(THRESHHOLD_VAL, float(torch.abs(accuracy_pitch)))
                c3 = operator.gt(THRESHHOLD_VAL, float(torch.abs(accuracy_roll)))
                if (c1 & c2 & c3) == True:
                    eval_acc += 1
        # Save models at numbered epochs.
        print(' Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, Acc: %.6f'
              % (eval_loss_yaw / len_val_data,
                 eval_loss_pitch / len_val_data, eval_loss_roll / len_val_data, train_acc / (BATCH_SIZE * n)))
        print('Finished Training')
        print('gpu_id:', gpu_id)
        print(model_save_name)
        torch.save(model.state_dict(),
                   model_save_name)

        writer.add_scalar('Train/Loss_yaw', eval_loss_yaw / len_val_data, epoch)
        writer.add_scalar('Train/Loss_pitch', eval_loss_pitch / len_val_data, epoch)
        writer.add_scalar('Train/Loss_roll', eval_loss_roll / len_val_data, epoch)
        writer.add_scalar('Train/Accuracy', eval_acc / len_val_data, epoch)


def testing_processing(checkpoint_path, model, test_loader, loss_func, test_data, THRESHHOLD_TRAIN, result_mat_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    test_loss = 0.
    test_acc = 0.
    index_all = []
    out_all = []
    label_all = []
    for index, batch_image, batch_label1, batch_label2, batch_label3 in test_loader:
        # print(index)
        batch_image, batch_label1, batch_label2, batch_label3 = Variable(batch_image.cuda(),
                                                                         volatile=True), Variable(
            batch_label1.cuda(), volatile=True), Variable(batch_label2.cuda(), volatile=True), Variable(
            batch_label3.cuda(), volatile=True),
        out = model(batch_image)
        batch_label1 = batch_label1.type(torch.cuda.FloatTensor)
        batch_label2 = batch_label2.type(torch.cuda.FloatTensor)
        batch_label3 = batch_label3.type(torch.cuda.FloatTensor)
        batch_label = torch.stack((batch_label1, batch_label2, batch_label3), dim=1)
        loss = loss_func(out, batch_label)
        test_loss += loss.data[0]
        for i in range(batch_image.shape[0]):
            accuracy = torch.add(out[i], -batch_label[i])
            c = [operator.gt(THRESHHOLD_TRAIN, float(i)) for i in torch.abs(accuracy)]
            if (c[0] & c[1] & c[2]) == True:
                test_acc += 1
        out_cur = out.cpu().data.numpy()
        out_all.extend(out_cur)
        batch_label_cur = batch_label.cpu().data.numpy()
        label_all.extend(batch_label_cur)
        index_all.extend(index)

        print(index)

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / (len(test_data)), test_acc / (len(test_data))))
    print('Finished Testing')

    mat_data = {'Alex_net_out_image_path': index_all, 'Alex_net_output_rot': out_all,
                'Alex_net_input_rot': label_all}
    sio.savemat(result_mat_path, mat_data)
