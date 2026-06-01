import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from storage.gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh2")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)

    # NEW: Patch parameters
    parser.add_argument("--patch_size", type=int, default=24, help="patch size")
    parser.add_argument("--stride", type=int, default=24, help="stride for patching")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(0, os.cpu_count()))
    return parser.parse_args()


def get_dataset(data_path, flag, input_len, output_len):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)


def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Calculate number of patches
    num_patches = (args.input_len - args.patch_size) // args.stride + 1
    print(f"Number of patches: {num_patches}")
    print(f"Input length: {args.input_len}, Patch size: {args.patch_size}, Stride: {args.stride}")

    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                           num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                          num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device,
        input_len=args.input_len,
        patch_size=args.patch_size,
        stride=args.stride,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)

    # Update save path to include patch parameters
    save_path = f"./Embeddings/{args.data_path}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    print(f"Saving embeddings to: {save_path}")
    print(f"Expected embedding shape: [B, {args.d_model}, {args.num_nodes}, {num_patches}]")

    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        if i % 100 == 0:
            print(f"Processing batch {i}...")

        # Generate embeddings for all patches
        # Input: x [B, L, N], x_mark [B, L, mark_dim]
        # Output: embeddings [B, d_model, N, num_patches]
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))

        # Save embeddings
        file_path = f"{save_path}{i}.h5"
        with h5py.File(file_path, 'w') as hf:
            emb_np = embeddings.cpu().numpy()

            # 如果 batch size 为 1，就去掉 batch 维度
            if emb_np.shape[0] == 1:
                emb_np = emb_np.squeeze(0)  # (768, N, num_patches)
            #print(f"Saving squeezed embedding shape: {emb_np.shape}")
            else:
                print(f"Saving embedding shape with batch: {emb_np.shape}")

            hf.create_dataset('embeddings', data=emb_np)
            hf.create_dataset('num_patches', data=num_patches)
            hf.create_dataset('patch_size', data=args.patch_size)
            hf.create_dataset('stride', data=args.stride)

        # Print shape for first batch to verify
        if i == 0:
            print(f"First batch embedding shape: {embeddings.shape}")
            print(f"Expected: [{args.batch_size}, {args.d_model}, {args.num_nodes}, {num_patches}]")

    print(f"Embeddings saved successfully for {i + 1} batches")


if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1) / 60:.4f} minutes")