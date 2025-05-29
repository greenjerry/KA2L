from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self):
        self.data = list()
        self.ids_to_idx = dict()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def append(self, *, id, prompt, question, generated_text, full_answers, hidden_last_tok_before_gen, ground_truth,
               log_likelihood=None, mean_pooled_embedding=None):
        if log_likelihood is None:
            log_likelihood = []
        self.data.append({
            'id': id,
            'prompt': prompt,
            'question': question,
            'generated_text': generated_text,
            'full_answers': full_answers,
            'hidden_last_tok_before_gen': hidden_last_tok_before_gen,
            'mean_pooled_embedding': mean_pooled_embedding,
            'ground_truth': ground_truth,
            'log_likelihood': log_likelihood
        })
        self.ids_to_idx[id] = len(self.data) - 1

    def del_hidden_states(self):
        for idx in range(len(self.data)):
            if 'hidden_last_tok_before_gen' in self.data[idx]:
                del self.data[idx]['hidden_last_tok_before_gen']
            if 'mean_pooled_embedding' in self.data[idx]:
                del self.data[idx]['mean_pooled_embedding']
