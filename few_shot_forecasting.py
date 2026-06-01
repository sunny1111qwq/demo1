#!/usr/bin/env python
# coding: utf-8

import argparse, time, os, random, faulthandler, torch, csv
import torch.nn as nn
from types import SimpleNamespace
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time import TriModal
from utils.metrics import MSE, MAE, metric
import multiprocessing

faulthandler.enable()


class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        lrate,
        wdecay,
        device,
        epochs
    ):
        self.model = TriModal(
            device=device,
            channel=channel,
            num_nodes=num_nodes,
            seq_len=seq_len,
            pred_len=pred_len,
            dropout_n=dropout_n,
            d_llm=d_llm,
            e_layer=e_layer,
            d_layer=d_layer,
            head=head
        )

        self.epochs = epochs
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lrate,
            weight_decay=wdecay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=min(epochs, 50),
            eta_min=1e-6
        )
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5

        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()

        predict = self.model(input, mark, embeddings)
        loss = self.loss(predict, real)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mae = self.MAE(predict, real)
        return loss.item(), mae.item()

    def eval(self, input, mark, embeddings, real_val):
        self.model.eval()

        with torch.no_grad():
            predict = self.model(input, mark, embeddings)

        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()


def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }

    data_class = data_map.get(args.data_path, Dataset_Custom)

    train_set = data_class(
        flag='train',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path
    )
    val_set = data_class(
        flag='val',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path
    )
    test_set = data_class(
        flag='test',
        scale=True,
        size=[args.seq_len, 0, args.pred_len],
        data_path=args.data_path
    )

    scaler = train_set.scaler

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_experiment_results(args, mse, mae, save_file="experiment_results.few_shot.csv"):
    experiment_name = (
        f"{args.data_path}_sl{args.seq_len}_pl{args.pred_len}_"
        f"ch{args.channel}_bs{args.batch_size}_ps{args.patch_size}_"
        f"lr{args.learning_rate}_dn{args.dropout_n}_dllm{args.d_llm}_"
        f"el{args.e_layer}_dl{args.d_layer}_h{args.head}_"
        f"wd{args.weight_decay}_ep{args.epochs}_seed{args.seed}"
    )

    result_string = f"MSE: {mse:.6f}, MAE: {mae:.6f}"
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file)

    with open(result_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([experiment_name, result_string])

    print(f"\nExperiment results saved to {result_path}")
    print(experiment_name)
    print(result_string)


def main():
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

    args = SimpleNamespace(
        device='cuda',
        data_path='ETTh2',
        channel=128,
        batch_size=64,
        patch_size=24,
        dropout_n=0.5,
        e_layer=1,
        d_layer=3,
        num_nodes=7,
        seq_len=96,
        pred_len=192,
        learning_rate=1e-3,
        d_llm=768,
        head=8,
        weight_decay=1e-3,
        num_workers=0,
        model_name='gpt2',
        epochs=40,
        seed=2024,
        es_patience=10,
        save='./logs/custom-save-path-',
        few_shot_ratio=0.1,
        few_shot_mode='random'
    )

    seed_everything(args.seed)

    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    num_train = len(train_set)
    few_shot_size = int(num_train * args.few_shot_ratio)
    indices = list(range(num_train))
    g1 = torch.Generator().manual_seed(args.seed)

    if args.few_shot_mode == 'last':
        subset_indices = list(range(num_train - few_shot_size, num_train))
        few_shot_subset = Subset(train_set, subset_indices)
    elif args.few_shot_mode == 'first':
        subset_indices = indices[:few_shot_size]
        few_shot_subset = Subset(train_set, subset_indices)
    elif args.few_shot_mode == 'random':
        few_shot_subset, _ = random_split(
            train_set,
            [few_shot_size, num_train - few_shot_size],
            generator=g1
        )
    else:
        raise ValueError(
            f"Unsupported few_shot_mode: {args.few_shot_mode}. "
            "Choose from ['last', 'first', 'random']."
        )

    g2 = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        few_shot_subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=g2,
        worker_init_fn=seed_worker
    )

    print(f"Data ready for few shot learning ({args.few_shot_mode} {args.few_shot_ratio:.0%} of train set)")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0
    bestid = 0

    path = os.path.join(
        args.save,
        args.data_path,
        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}/"
    )

    if not os.path.exists(path):
        os.makedirs(path)

    his_train_loss = []
    his_train_mae = []
    his_loss = []
    his_loss_mae = []
    val_time = []
    train_time = []

    print(args)

    engine = trainer(
        scaler=scaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs
    )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mae = []

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainx_mark = torch.Tensor(x_mark).to(device)
            train_embedding = torch.Tensor(embeddings).to(device)

            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)

            train_loss.append(metrics[0])
            train_mae.append(metrics[1])

        t2 = time.time()
        print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, t2 - t1))
        train_time.append(t2 - t1)

        val_loss = []
        val_mae = []
        s1 = time.time()

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(val_loader):
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)
            valx_mark = torch.Tensor(x_mark).to(device)
            val_embedding = torch.Tensor(embeddings).to(device)

            metrics = engine.eval(valx, valx_mark, val_embedding, valy)

            val_loss.append(metrics[0])
            val_mae.append(metrics[1])

        s2 = time.time()
        print("Epoch: {:03d}, Validation Time: {:.4f} secs".format(i, s2 - s1))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)

        his_train_loss.append(mtrain_loss)
        his_train_mae.append(mtrain_mae)
        his_loss.append(mvalid_loss)
        his_loss_mae.append(mvalid_mae)

        print("-----------------------")
        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}".format(
                i,
                mtrain_loss,
                mtrain_mae
            ),
            flush=True
        )
        print(
            "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}".format(
                i,
                mvalid_loss,
                mvalid_mae
            ),
            flush=True
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")

            if i <= 10:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mse = 0

                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)

            else:
                test_outputs = []
                test_y = []

                for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
                    testx = torch.Tensor(x).to(device)
                    testy = torch.Tensor(y).to(device)
                    testx_mark = torch.Tensor(x_mark).to(device)
                    test_embedding = torch.Tensor(embeddings).to(device)

                    with torch.no_grad():
                        preds = engine.model(testx, testx_mark, test_embedding)

                    test_outputs.append(preds)
                    test_y.append(testy)

                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y, dim=0)

                amse = []
                amae = []

                for j in range(args.pred_len):
                    pred = test_pre[:, j, ].to(device)
                    real = test_real[:, j, ].to(device)
                    metrics = metric(pred, real)

                    amse.append(metrics[0])
                    amae.append(metrics[1])

                print(
                    "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}".format(
                        np.mean(amse),
                        np.mean(amae)
                    )
                )

                if np.mean(amse) < test_log:
                    test_log = np.mean(amse)
                    loss = mvalid_loss

                    torch.save(engine.model.state_dict(), path + "best_model.pth")

                    epochs_since_best_mse = 0
                    bestid = i

                    print("Test low! Updating! Test Loss: {:.4f}".format(np.mean(amse)), end=", ")
                    print("Test low! Updating! Valid Loss: {:.4f}".format(mvalid_loss), end=", ")
                    print("epoch: ", i)

                else:
                    epochs_since_best_mse += 1
                    print("No update")

        else:
            epochs_since_best_mse += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best_mse >= args.es_patience and i >= args.epochs // 2:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)

    if bestid > 0:
        print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
    else:
        print("No best model was saved.")
        return

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))

    test_outputs = []
    test_y = []

    for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        testx_mark = torch.Tensor(x_mark).to(device)
        test_embedding = torch.Tensor(embeddings).to(device)

        with torch.no_grad():
            preds = engine.model(testx, testx_mark, test_embedding)

        test_outputs.append(preds)
        test_y.append(testy)

    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.cat(test_y, dim=0)

    amse = []
    amae = []

    for j in range(args.pred_len):
        pred = test_pre[:, j, ].to(device)
        real = test_real[:, j, ].to(device)
        metrics = metric(pred, real)

        amse.append(metrics[0])
        amae.append(metrics[1])

    print(
        "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}".format(
            np.mean(amse),
            np.mean(amae)
        )
    )


    final_mse = float(np.mean(amse))
    final_mae = float(np.mean(amae))
    save_experiment_results(args, final_mse, final_mae)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()