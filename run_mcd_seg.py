import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from lib.mcd_seg import MCD_seg
from lib.utils.torch_utils import adjust_learning_rate
from lib.utils.fvi_seg_utils import test, numpy_metrics, run_runtime_seg
from lib.utils.camvid import get_camvid

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='mcd', help='Choose from mcd or deterministic')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_size_ft', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--final_epoch', type=int, default=1001)
parser.add_argument('--ft_start', type=int, default=1000, help='When to start fine-tuning on full-size images')
parser.add_argument('--dataset', type=str, default='camvid', help='camvid')
parser.add_argument('--save_results', type=int, default=500, help='save results every few epochs')
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

if args.dataset == 'camvid':
	train_loader, val_loader, test_loader, ft_loader, num_classes = get_camvid(args.batch_size, args.batch_size_ft)

exp_name = '{}_segmentation_{}'.format(args.dataset, args.model_type)


from lib.elbo_seg import masked_cross_entropy_seg as loss_seg
print('Training with masked Cross-Entropy loss + Re-scaled boltzmann distribution')

def train(num_epochs, train_loader):
	model.train()

	ft_start_flag = 0
	for s in range(args.start_epoch, args.start_epoch + args.n_epochs + 1):
		train_loss = 0.
		train_error = 0.
		model.train()
		if s >= args.ft_start and ft_start_flag == 0:
			print('Now fine-tuning on full size images')
			train_loader = ft_loader
			ft_start_flag += 1
		for X, Y in tqdm(train_loader):
			x_t = X.to(device)
			y_t = Y.to(device)
			N_t = x_t.size(0)

			optimizer.zero_grad()

			mean, logvar_aleatoric = model(x_t)

			rescaled_logits = mean * torch.exp(-logvar_aleatoric)
			loss = - loss_seg(y_t, rescaled_logits, print_loss=True)

			loss.backward()
			train_loss += -loss.item()
			optimizer.step()
			_, _, train_acc_curr = numpy_metrics(rescaled_logits.argmax(1).data.cpu().view(N_t, -1).numpy(), y_t.view(N_t, -1).data.cpu().numpy())
			train_error += 1 - train_acc_curr
			adjust_learning_rate(args.lr, 0.998, optimizer, s, args.final_epoch)
			del x_t, y_t, mean, logvar_aleatoric
		train_loss /= len(train_loader)
		train_error /= len(train_loader)
		print('Epoch: {} || Average Train Error: {:.5f} || Average Train Loss: {:.5f}'.format(s, train_error, train_loss))
		np.savetxt('{}_{}_epoch_{}_average_train_loss.txt'.format(args.dataset, exp_name, s), [train_loss])
		np.savetxt('{}_{}_epoch_{}_average_train_error.txt'.format(args.dataset, exp_name, s), [train_error])

		if s % args.save_results == 0 or s == args.final_epoch:
			val_error, val_mIOU = test(model, val_loader, num_classes, args.dataset, exp_name, plot_imgs=False)
			print('Epoch: {} || Validation Error: {:.5f} || Validation Mean IOU: {:.5f}'.format(s, val_error, val_mIOU))
			torch.save(model.state_dict(), 'model_{}_{}.bin'.format(args.dataset, exp_name))
			torch.save(optimizer.state_dict(), 'optimizer_{}_{}.bin'.format(args.dataset, exp_name))
			np.savetxt('{}_{}_epoch_{}_val_error.txt'.format(args.dataset, exp_name, s), [val_error])
			np.savetxt('{}_{}_epoch_{}_val_mIOU.txt'.format(args.dataset, exp_name, s), [val_mIOU])

if __name__ == '__main__':

	device = torch.device("cuda")

	model = MCD_seg(num_classes, args.model_type)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

	if args.load:
		model_load_dir = os.path.join(args.base_dir, 'FVI_CV/model_{}_{}.bin'.format(args.dataset, exp_name))
		optimizer_load_dir = os.path.join(args.base_dir, 'FVI_CV/optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 
		model.load_state_dict(torch.load(model_load_dir))
		optimizer.load_state_dict(torch.load(optimizer_load_dir))
		print('Loading deterministic segmentation model..')

	if args.training_mode:
		print('Training determinstic segmentation for {} epochs'.format(args.n_epochs))
		train(args.n_epochs, train_loader)
	if args.test_mode:
		print('Evaluating {} segmentation on test set'.format(args.model_type))
		model_load_dir = os.path.join(args.base_dir, 'FVI_CV/models_test/model_{}_mcd_seg_test.bin'.format(args.dataset))
		model.load_state_dict(torch.load(model_load_dir))
		error, mIOU = test(model, test_loader, num_classes, args.dataset, exp_name, plot_imgs=True, mkdir=True)
		print('Test Accuracy: {:.5f} || Test Mean IOU: {:.5f}'.format(1. - error, mIOU))
		np.savetxt('{}_{}_epoch_{}_test_accuracy.txt'.format(args.dataset, exp_name, -1), [1. - error])
		np.savetxt('{}_{}_epoch_{}_test_mIOU.txt'.format(args.dataset, exp_name, -1), [mIOU])
	if args.test_runtime_mode:
		model_load_dir = os.path.join(args.base_dir, 'FVI_CV/models_test/model_{}_mcd_seg_test.bin'.format(args.dataset))
		model.load_state_dict(torch.load(model_load_dir))
		run_runtime_seg(model, test_loader, exp_name, 50)
