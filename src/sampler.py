import numpy as np


class BalancedSampler:
    def __init__(self, targets, batch_size, num_batches, seed=42):
        assert batch_size % 8 == 0, "batch_size must be divisible by 8"
        self.targets = np.array(targets)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.num_classes = 8
        
        self.class_indices = {
            c: np.where(self.targets == c)[0]
            for c in range(self.num_classes)
        }
    
    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        
        for _ in range(self.num_batches):
            per_class = self.batch_size // self.num_classes
            
            batch = []
            for c in range(self.num_classes):
                idx = self.class_indices[c]
                chosen = rng.choice(idx, size=per_class, replace=True)
                batch.extend(chosen)
            
            batch = np.array(batch)
            rng.shuffle(batch)
            yield batch.tolist()
    
    def __len__(self):
        return self.num_batches
