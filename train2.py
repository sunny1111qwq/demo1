import torch
from torch import optim
import torch.nn.functional as F
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
import random
import csv
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from models.T3Time import TriModal
from utils.metrics import MSE, MAE, metric
import faulthandler

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1", help="data path")
    parser.add_argument("--channel", type=int, default=128, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=96, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=96, help="out_len")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--patch_size", type=int, default=24, help="patch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.4, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--d_ff", type=int, default=32, help="feed-forward dimension in CrossModal")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=4, help="heads of attention")
    parser.add_argument("--num_cma_heads", type=int, default=4, help="number of independent CMA modules")
    parser.add_argument("--cma_n_heads", type=int, default=1, help="attention heads inside each CMA module")
    parser.add_argument("--cma_gate_hidden", type=int, default=128, help="hidden dimension of CMA head fusion gate")
    parser.add_argument("--vision_mid", type=int, default=-1, help="hidden channels of vision CNN, -1 uses max(channel // 4, 32)")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument("--es_patience", type=int, default=5, help="quit if no improvement after this many iterations")
    parser.add_argument("--save", type=str, default="./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "-",
                        help="save path")

    # ========== HybridMemory Hyperparameters ==========
    parser.add_argument("--mem_num", type=int, default=50,
                        help="number of learnable memory slots (10-30 recommended)")
    parser.add_argument("--mem_dim", type=int, default=64,
                        help="dimension of learnable memory (32-128 recommended)")
    parser.add_argument("--dynamic_mem_size", type=int, default=50,
                        help="size of dynamic memory bank (50-200 recommended)")
    parser.add_argument("--mem_top_k", type=int, default=5,
                        help="top-k for dynamic memory retrieval (3-10 recommended)")
    return parser.parse_args()


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
            d_ff,

            e_layer,
            d_layer,
            head,
            num_cma_heads,
            cma_n_heads,
            cma_gate_hidden,
            vision_mid,






            lrate,
            wdecay,
            device,
            epochs,
            mem_num,
            mem_dim,
            dynamic_mem_size,
            mem_top_k
    ):
        self.model = TriModal(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len,
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, d_layer=d_layer, d_ff=d_ff, head=head,
            num_cma_heads=num_cma_heads, cma_n_heads=cma_n_heads,
            cma_gate_hidden=cma_gate_hidden, vision_mid=vision_mid,




            mem_num=mem_num, mem_dim=mem_dim, dynamic_mem_size=dynamic_mem_size, mem_top_k=mem_top_k
        )
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.loss = MSE
        self.MAE = MAE
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))

    def train(self, input, mark, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input, mark, embeddings)
        pred_loss = self.loss(predict, real)
        loss = pred_loss
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
        pred_loss = self.loss(predict, real_val)
        loss = pred_loss
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()

    def set_epoch(self, epoch_id):
        if hasattr(self.model, "set_epoch_ratio"):
            self.model.set_epoch_ratio(epoch_id)


def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    data_class = data_map.get(args.data_path, Dataset_Custom)
    train_set = data_class(flag='train', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)
    val_set = data_class(flag='val', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)

    scaler = train_set.scaler

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
                             num_workers=args.num_workers)
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


def save_experiment_results(args, mse, mae, save_file="experiment_results.csv"):
    """
    保存实验配置和结果到CSV文件

    Args:
        args: 命令行参数
        mse: 测试MSE结果
        mae: 测试MAE结果
        save_file: 保存文件名
    """
    # 生成实验名称
    experiment_name = (f"{args.data_path}_sl{args.seq_len}_pl{args.pred_len}_"
                       f"ch{args.channel}_bs{args.batch_size}_ps{args.patch_size}_"
                       f"lr{args.learning_rate}_dn{args.dropout_n}_dllm{args.d_llm}_dff{args.d_ff}_"
                       f"el{args.e_layer}_dl{args.d_layer}_h{args.head}_"
                       f"cmah{args.num_cma_heads}_cmanh{args.cma_n_heads}_cmagh{args.cma_gate_hidden}_vm{args.vision_mid}_"



                       f"wd{args.weight_decay}_ep{args.epochs}_seed{args.seed}_"
                       f"mtk{args.mem_top_k}")

    # 格式化结果字符串
    result_string = f"MSE: {mse:.6f}, MAE: {mae:.6f}"

    # 准备要保存的数据（只保存实验名称和结果）
    result_data = {
        'experiment_name': experiment_name,
        'results': result_string
    }

    save_file = os.path.abspath(save_file)

    # 检查文件是否存在
    file_exists = os.path.isfile(save_file)

    # 写入CSV文件
    with open(save_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['experiment_name', 'results']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()

        # 写入数据行
        writer.writerow(result_data)

    print(f"\nExperiment results saved to {save_file}")
    print(f"{experiment_name}")
    print(f"{result_string}")


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_csv_path = os.path.join(project_root, "experiment_results.csv")

    # 设置随机种子（在加载数据之前）
    seed_it(args.seed)

    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

    print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0

    path = os.path.join(args.save, args.data_path,
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}/")
    if not os.path.exists(path):
        os.makedirs(path)

    his_loss = []
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
        d_ff=args.d_ff,

        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        num_cma_heads=args.num_cma_heads,
        cma_n_heads=args.cma_n_heads,
        cma_gate_hidden=args.cma_gate_hidden,
        vision_mid=args.vision_mid,






        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        mem_num=args.mem_num,
        mem_dim=args.mem_dim,
        dynamic_mem_size=args.dynamic_mem_size,
        mem_top_k=args.mem_top_k
    )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        engine.set_epoch(i)

        t1 = time.time()
        train_loss = []
        train_mae = []

        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(train_loader):
            trainx = torch.Tensor(x).to(device)  # [B, L, N]
            trainy = torch.Tensor(y).to(device)
            trainx_mark = torch.Tensor(x_mark).to(device)
            train_embedding = torch.Tensor(embeddings).to(device)
            metrics = engine.train(trainx, trainx_mark, train_embedding, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
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
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} "
        print(
            log.format(i, mtrain_loss, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mae),
            flush=True,
        )

        if mvalid_loss < loss:
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            bestid = i
            epochs_since_best_mse = 0
            print(f'Epoch {i}: Validation loss improved to {mvalid_loss:.4f}, model saved.')
        else:
            epochs_since_best_mse += 1
            print(f'Epoch {i}: No improvement. Best was epoch {bestid} with loss {loss:.4f}')

            # 早停检查
        if epochs_since_best_mse >= args.es_patience:
            print(f'Early stopping triggered after {i} epochs. Best epoch was {bestid}.')
            break

        # Step LR scheduler once per epoch.
        engine.scheduler.step()

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

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
        log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
        amse.append(metrics[0])
        amae.append(metrics[1])

    log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
    print(log.format(np.mean(amse), np.mean(amae)))

    # 保存实验结果
    save_experiment_results(args, np.mean(amse), np.mean(amae), save_file=results_csv_path)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))