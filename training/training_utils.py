from utils import evaluation


class BaseTrainingPipeline():
    def train_epoch(self, model, loss_fn, optimizer, train_dl):
        loss_sum = 0.0
        ctr = 0
        for x_batch, y_batch, next_batch in train_dl:
            model.train()
            pred_next = model(x_batch)
            loss = loss_fn(next_batch, pred_next)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train = loss.item()
            loss_sum += loss_train
            ctr += 1
        return float(loss_sum / ctr)

    def train(self, model, loss, optimizer, train_dl, test_dl, num_epochs):
        for epoch in range(num_epochs):
            epoch_train_loss = BaseTrainingPipeline.train_epoch(model, loss, optimizer, train_dl)
            print("Training Loss: {0} - Epoch: {1}".format(epoch_train_loss, epoch + 1))
            test_acc = self.evaluate(model, test_dl)
            print("Testing Acc: {0} - Epoch: {1}".format(test_acc, epoch + 1))

    def evaluate(self, model, test_dl):
        model.eval()
        total_acc = total_em = total_mv = 0
        ctr = 0
        for X, Y, P in test_dl:
            pred = model(X).detach()
            anomaly_predict = Y.squeeze() #torch.tensor(anomaly_detector(pred.numpy(), P.numpy()))
            anomaly_label = Y.squeeze()
            acc = evaluation.accuracy(anomaly_predict, anomaly_label)
            total_acc += acc
            ctr += 1
            #scores = eval.torch_emmv_scores(self.model, X, scoring_func=scoring_function)
            #total_em += scores['em']
            #total_mv += scores['mv']
        print(total_acc / ctr, total_em / ctr, total_mv / ctr)
        return total_acc / ctr, total_em / ctr, total_mv / ctr
