from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.MCLP.state_MCLP import StateMCLP



class MCLP(object):
    NAME = 'MCLP'

    @staticmethod
    def get_total_num(dataset, pi):
        users = dataset['users']
        facilities = dataset['facilities']
        demand = dataset['demand']
        radius = dataset['r'][0]
        batch_size, n_user, _ = users.size()
        _, n_facilities, _ = facilities.size()
        _, p = pi.size()
        dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
        facility_tensor = pi.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_user))
        f_u_dist_tensor = dist.gather(1, facility_tensor)
        mask = f_u_dist_tensor < radius
        cover_num = torch.sum(torch.mul(demand.squeeze(-1), mask.any(dim=1).float()), dim=-1)
        # cover_num = torch.as_tensor(cover_num, dtype=torch.float32)

        return cover_num

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCLPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMCLP.initialize(*args, **kwargs)


class MCLPDataset(Dataset):
    def __init__(self, filename=None, n_users=50, n_facilities=20, num_samples=5000, offset=0, p=8, r=0.2, distribution=None):
        super(MCLPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [row for row in (data[offset:offset + num_samples])]
                p = self.data[0]['p']
                r = self.data[0]['r']
        else:
            # Sample points randomly in [0, 1] square
            self.data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                              facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                              demand=torch.FloatTensor(n_users, 1).uniform_(1, 10),
                              p=p,
                              r=r)
                         for i in range(num_samples)]

        self.size = len(self.data)
        self.p = p
        self.r = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]