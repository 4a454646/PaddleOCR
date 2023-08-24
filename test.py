from paddle.io import RandomSampler, BatchSampler, Dataset

# init with dataset
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

bs = BatchSampler(dataset=RandomDataset(100),
                  shuffle=False,
                  batch_size=16,
                  drop_last=False)

for batch_indices in bs:
    print(batch_indices)

# init with sampler
sampler = RandomSampler(RandomDataset(100))
bs = BatchSampler(sampler=sampler,
                  batch_size=8,
                  drop_last=True)

for batch_indices in bs:
    print(batch_indices)
