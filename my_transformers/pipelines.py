import torch
import torch.nn.functional as F
def pipeline(mode, model, tokenizer, device="cpu"):
    if mode == 'text-generation':
        def generate(prompt, max_len=1024):
            next_token = torch.tensor([411])
            model.eval()
            idx = tokenizer.encode("<s>"+ prompt).ids
            idx = torch.tensor(idx).unsqueeze(0).to(device)
            while next_token != tokenizer.encode("</s>").ids[0] and len(idx[0]) < max_len:
                with torch.no_grad():
                    logits, _ = model(idx[:, -model.config.block_size:])
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)

                    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                    next_token = torch.multinomial(topk_probs, num_samples=1)
                    next_token = torch.gather(topk_indices, -1, next_token)
                    idx = torch.cat((idx, next_token), dim=1)
            return tokenizer.decode(idx[0].tolist()[1:-1])
    return generate

    
    