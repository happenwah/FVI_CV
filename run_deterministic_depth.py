import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from lib.model.densenet import FCDenseNet_103
from lib.utils.torch_utils import adjust_learning_rate
from lib.utils.deterministic_depth_utils import run_test_deterministic, run_runtime_deterministic

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--p', type=float, default=0.2)
parser.add_argument('--dataset', type=str, default='make3d')
parser.add_argument('--loss', type=str, default='l1', help='choose from l1 or berhu')
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=4000)
parser.add_argument('--save_results', type=int, default=500, help='save results every x epochs')
parser.add_argument('--base_dir', type=str, default='/rdsgpfs/general/user/etc15/home/', help='directory in which datasets are contained')
parser.add_argument('--training_mode', action='store_true')
parser.add_argument('--load', action='store_true', help='Load model for resuming training, default: False')
parser.add_argument('--test_mode', action='store_true')
parser.add_argument('--test_runtime_mode', action='store_true')

args = parser.parse_args()

if args.training_mode:
	torch.backends.cudnn.benchmark = True
elif args.test_runtime_mode:
	torch.backends.cudnn.deterministic = True

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed(args.randseed)

if args.dataset == 'make3d':
	from lib.utils.make3d_loader import Make3dDataset
	H, W = 168, 224
	dir_train = os.path.join(args.base_dir, 'datasets/make3d/make3d_train.npz')
	dir_test = os.path.join(args.base_dir, 'datasets/make3d/make3d_test.npz')
	training_set_full_size = Make3dDataset(train=True, dir=dir_train)
	test_set = Make3dDataset(train=False, dir=dir_test)

exp_name = '{}_fcdensenet103_deterministic_loss_{}'.format(args.dataset, args.loss)

if args.loss == 'berhu':
	from lib.utils.deterministic_depth_utils import berhu_loss
	print('Using berHu loss')
elif args.loss == 'l1':
	print('Using L1 loss')

def train(num_epochs):
	model.train()
	params = {'batch_size': args.batch_size,
                        'shuffle': True,
                        'num_workers': 0}

	N_train = training_set_full_size.__len__()
	training_generator = torch.utils.data.DataLoader(training_set_full_size, **params)

	N_test = test_set.__len__()

	for s in range(args.n_epochs + 1):
		pred_list = []
		target_list = []
		mask_list = []
		for X, Y in tqdm(training_generator):
			x_t = X.to(device)
			y_t = Y.to(device)

			if args.dataset == 'make3d':
				mask = (y_t < 1.0)
                                   
			if mask.long().sum() > 1000.:
				optimizer.zero_grad()                                                   
				mean = model(x_t)
				N = x_t.size(0)

				if args.loss == 'l1':
					loss = torch.abs(y_t.view(N, -1) - mean.view(N, -1))[mask.view(N, -1)].mean(0)
				elif args.loss == 'berhu':
					loss = berhu_loss(y_t.view(N, -1), mean.view(N, -1), mask.view(N, -1))

				loss.backward()
				pred_list.append(mean.view(N, -1).cpu().detach())
				target_list.append(y_t.view(N, -1).cpu().detach())
				mask_list.append(mask.view(N, -1).cpu().detach())
				#clip gradients
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
				optimizer.step()
			else:
				continue
			del y_t, mean, mask, x_t
		train_masks = torch.cat(mask_list, 0)
		train_preds = torch.cat(pred_list, 0)[train_masks]
		train_targets = torch.cat(target_list, 0)[train_masks]
		train_mse = torch.pow(train_preds - train_targets, 2).mean().item()
		if args.dataset=='make3d':
			train_rmse = (70.) * (train_mse ** 0.5)
		print('Epoch: {} || Train RMSE: {:.5f}'.format(s, train_rmse))
		np.savetxt('{}_{}_epoch_{}_train_rmse.txt'.format(args.dataset, exp_name, s), [train_rmse])
		del train_masks, train_preds, train_targets

		if s % args.save_results == 0:
			run_test_deterministic(s, model, test_set, N_test, args.dataset, exp_name)
			torch.save(model.state_dict(), 'model_{}_{}.bin'.format(args.dataset, exp_name))
			torch.save(optimizer.state_dict(), 'optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 

if __name__ == '__main__':
	device = torch.device("cuda")

	model = FCDenseNet_103().to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

	if args.load:
		load_dir_model = os.path.join(args.base_dir, 'FVI/model_{}_{}.bin'.format(args.dataset, exp_name))
		load_dir_optimizer = os.path.join(args.base_dir,'FVI/optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 
		model.load_state_dict(torch.load(load_dir_model))
		optimizer.load_state_dict(torch.load(load_dir_optimizer))
		print('Loading FCDensenet 103 model..')

	if args.training_mode:
		train(args.n_epochs)
	if args.test_mode:
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_deterministic_{}_test.bin'.format(args.loss))
		model.load_state_dict(torch.load(load_dir_model))
		N_test = test_set.__len__()
		run_test_deterministic(-1, model, test_set, N_test, args.dataset, exp_name)
	if args.test_runtime_mode:
		load_dir_model = os.path.join(args.base_dir, 'FVI/models_test/model_deterministic_{}_test.bin'.format(args.loss))
		model.load_state_dict(torch.load(load_dir_model))
		run_runtime_deterministic(model, test_set, exp_name)
