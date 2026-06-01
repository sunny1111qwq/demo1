import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model


class GenPromptEmb(nn.Module):
    def __init__(
            self,
            data_path='FRED',
            model_name="gpt2",
            device='cuda:0',
            input_len=96,
            patch_size=24,
            stride=24,
            d_model=768,
            layer=12,
            divide='train'
    ):
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len = input_len
        self.patch_size = patch_size
        self.stride = stride
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer

        # Calculate number of patches
        self.num_patches = (input_len - patch_size) // stride + 1

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    def _prepare_prompt_for_patch(self, input_template, patch_data, patch_data_mark, i, j, patch_idx):
        """
        Prepare prompt for a specific patch
        Args:
            input_template: Template string for the prompt
            patch_data: [B, patch_size, num_nodes] - data for current patch
            patch_data_mark: [B, patch_size, mark_dim] - time marks for current patch
            i: batch index
            j: node/feature index
            patch_idx: index of current patch
        """
        # Time series values for this patch
        values = patch_data[i, :, j].flatten().tolist()
        values_str = ", ".join([f"{value:.2f}" for value in values])

        # Trend within this patch
        trends = torch.sum(torch.diff(patch_data[i, :, j].flatten()))
        trends_str = f"{trends.item():.2f}"

        # Start and end dates for this patch
        start_idx = 0
        end_idx = self.patch_size - 1

        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(patch_data_mark[i, start_idx, 2]):02d}/{int(patch_data_mark[i, start_idx, 1]):02d}/{int(patch_data_mark[i, start_idx, 0]):04d}"
            end_date = f"{int(patch_data_mark[i, end_idx, 2]):02d}/{int(patch_data_mark[i, end_idx, 1]):02d}/{int(patch_data_mark[i, end_idx, 0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(patch_data_mark[i, start_idx, 2]):02d}/{int(patch_data_mark[i, start_idx, 1]):02d}/{int(patch_data_mark[i, start_idx, 0]):04d} {int(patch_data_mark[i, start_idx, 4]):02d}:00"
            end_date = f"{int(patch_data_mark[i, end_idx, 2]):02d}/{int(patch_data_mark[i, end_idx, 1]):02d}/{int(patch_data_mark[i, end_idx, 0]):04d} {int(patch_data_mark[i, end_idx, 4]):02d}:00"
        else:  # ETTm1, ETTm2, Weather
            start_date = f"{int(patch_data_mark[i, start_idx, 2]):02d}/{int(patch_data_mark[i, start_idx, 1]):02d}/{int(patch_data_mark[i, start_idx, 0]):04d} {int(patch_data_mark[i, start_idx, 4]):02d}:{int(patch_data_mark[i, start_idx, 5]):02d}"
            end_date = f"{int(patch_data_mark[i, end_idx, 2]):02d}/{int(patch_data_mark[i, end_idx, 1]):02d}/{int(patch_data_mark[i, end_idx, 0]):04d} {int(patch_data_mark[i, end_idx, 4]):02d}:{int(patch_data_mark[i, end_idx, 5]):02d}"

        # Create prompt for this patch
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        in_prompt = in_prompt.replace("PatchN", f"Patch{patch_idx + 1}")

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def create_patches(self, x, x_mark):
        """
        Create overlapping patches from input time series
        Input:
            x: [B, L, N] where L is seq_len
            x_mark: [B, L, mark_dim]
        Output:
            patches: [B, num_patches, patch_size, N]
            patches_mark: [B, num_patches, patch_size, mark_dim]
        """
        B, L, N = x.shape
        _, _, mark_dim = x_mark.shape
        patches = []
        patches_mark = []

        for i in range(self.num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_size
            patch = x[:, start_idx:end_idx, :]  # [B, patch_size, N]
            patch_mark = x_mark[:, start_idx:end_idx, :]  # [B, patch_size, mark_dim]
            patches.append(patch)
            patches_mark.append(patch_mark)

        patches = torch.stack(patches, dim=1)  # [B, num_patches, patch_size, N]
        patches_mark = torch.stack(patches_mark, dim=1)  # [B, num_patches, patch_size, mark_dim]

        return patches, patches_mark

    def generate_embeddings(self, in_data, in_data_mark):
        """
        Generate embeddings for all patches
        Args:
            in_data: [B, L, N]
            in_data_mark: [B, L, mark_dim]
        Returns:
            embeddings: [B, d_model, N, num_patches] - embeddings for each patch and node
        """
        input_templates = {
            'FRED': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
            'ILI': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
            'ETTh1': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTh2': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ECL': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTm1': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'ETTm2': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'Weather': "PatchN: From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
        }

        input_template = input_templates.get(self.data_path, input_templates['FRED'])

        # Create patches
        patches, patches_mark = self.create_patches(in_data, in_data_mark)
        # patches: [B, num_patches, patch_size, N]
        # patches_mark: [B, num_patches, patch_size, mark_dim]

        B, num_patches, patch_size, N = patches.shape

        # First pass: collect all tokenized prompts and find max length
        tokenized_prompts = []
        max_token_count = 0

        for i in range(B):
            for patch_idx in range(num_patches):
                for j in range(N):
                    patch_data = patches[:, patch_idx, :, :]  # [B, patch_size, N]
                    patch_mark = patches_mark[:, patch_idx, :, :]  # [B, patch_size, mark_dim]

                    tokenized_prompt = self._prepare_prompt_for_patch(
                        input_template, patch_data, patch_mark, i, j, patch_idx
                    ).to(self.device)

                    max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                    tokenized_prompts.append((i, patch_idx, j, tokenized_prompt))

        # Initialize embedding tensor: [B, max_token_count, d_model, N, num_patches]
        prompt_emb = torch.zeros(
            (B, max_token_count, self.d_model, N, num_patches),
            dtype=torch.float32,
            device=self.device
        )

        # Second pass: generate embeddings and pad
        for i, patch_idx, j, tokenized_prompt in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)

            # Pad if necessary
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                padding = last_token_embedding.repeat(1, padding_length, 1)
                prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
            else:
                prompt_embeddings_padded = prompt_embeddings

            prompt_emb[i, :, :, j, patch_idx] = prompt_embeddings_padded.squeeze(0)

        # Extract last token embeddings: [B, d_model, N, num_patches]
        last_token_emb = prompt_emb[:, -1, :, :, :]

        return last_token_emb