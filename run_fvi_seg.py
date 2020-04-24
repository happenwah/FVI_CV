import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from lib.fvi_seg import FVI_seg
from lib.utils.torch_utils import adjust_learning_rate
from lib.utils.fvi_seg_utils import test, numpy_metrics, run_runtime_seg
from lib.utils.camvid import get_camvid

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_size_ft', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--final_epoch', type=int, default=1001)
parser.add_argument('--ft_start', type=int, default=1000, help='When to start fine-tuning on full-size images')
parser.add_argument('--dataset', type=str, default='camvid', help='camvid')
parser.add_argument('--standard_cross_entropy', type=bool, default=False)
parser.add_argument('--add_cov_diag', type=bool, default=True, help='Add Diagonal component to Q covariance')
parser.add_argument('--f_prior', type=str, default='cnn_gp', help='Type of GP prior: cnn_gp')
parser.add_argument('--match_prior_mean', type=bool, default=False, help='Match Q mean with prior mean')
parser.add_argument('--x_inducing_var', type=float, default=0.1, help='Pixel-wise variance for inducing inputs')
parser.add_argument('--n_inducing', type=int, default=1, help='No. of inducing inputs, <= batch_size')
parser.add_argument('--save_results', type=int, default=1000, help='save results every few epochs')
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

if args.standard_cross_entropy:
	print('Using standard discrete likelihood')

if args.dataset == 'camvid':
	H_crop, W_crop, H, W = 224, 224, 360, 480
	train_loader, val_loader, test_loader, ft_loader, num_classes = get_camvid(args.batch_size, args.batch_size_ft)

if args.f_prior == 'cnn_gp':
	exp_name = '{}_segmentation_gp_bnn'.format(args.dataset)

from lib.elbo_seg import fELBO_seg as fELBO

def train(num_epochs, train_loader, FVI):
	FVI.train()

	ft_start_flag = 0
	for s in range(args.start_epoch, args.start_epoch + args.n_epochs + 1):
		train_loss = 0.
		train_error = 0.
		FVI.train()
		if s >= args.ft_start and ft_start_flag == 0:
			from lib.priors import f_prior_BNN
			print('Now fine-tuning on full size images')
			train_loader = ft_loader
			if args.f_prior == 'cnn_gp':
				FVI.prior = f_prior_BNN((H, W), device, num_channels_output=num_classes)
			ft_start_flag += 1
		for X, Y in tqdm(train_loader):
			x_t = X.to(device)
			y_t = Y.to(device)
			N_t = x_t.size(0)

			optimizer.zero_grad()

			f_samples, q_mean, q_cov, prior_mean, prior_cov = FVI(x_t)

			loss = - fELBO(y_t, f_samples, q_mean, q_cov, prior_mean, prior_cov, print_loss=True)

			loss.backward()
			train_loss += -loss.item()
			optimizer.step()
			_, _, train_acc_curr = numpy_metrics(FVI.predict(x_t, S=20).data.cpu().view(N_t, -1).numpy(), y_t.view(N_t, -1).data.cpu().numpy())
			train_error += 1 - train_acc_curr
			adjust_learning_rate(args.lr, 0.998, optimizer, s, args.final_epoch)
			del x_t, y_t, f_samples, q_mean, q_cov, prior_mean, prior_cov
		train_loss /= len(train_loader)
		train_error /= len(train_loader)
		print('Epoch: {} || Average Train Error: {:.5f} || Average Train Loss: {:.5f}'.format(s, train_error, train_loss))
		np.savetxt('{}_{}_epoch_{}_average_train_loss.txt'.format(args.dataset, exp_name, s), [train_loss])
		np.savetxt('{}_{}_epoch_{}_average_train_error.txt'.format(args.dataset, exp_name, s), [train_error])

		if s % args.save_results == 0 or s == args.final_epoch:
			val_error, val_mIOU = test(FVI, val_loader, num_classes, args.dataset, exp_name, plot_imgs=False)
			print('Epoch: {} || Validation Error: {:.5f} || Validation Mean IOU: {:.5f}'.format(s, val_error, val_mIOU))
			torch.save(FVI.state_dict(), 'model_{}_{}.bin'.format(args.dataset, exp_name))
			torch.save(optimizer.state_dict(), 'optimizer_{}_{}.bin'.format(args.dataset, exp_name))
			np.savetxt('{}_{}_epoch_{}_val_error.txt'.format(args.dataset, exp_name, s), [val_error])
			np.savetxt('{}_{}_epoch_{}_val_mIOU.txt'.format(args.dataset, exp_name, s), [val_mIOU])

if __name__ == '__main__':

	device = torch.device("cuda")

	keys = ('device', 'x_inducing_var', 'f_prior', 'n_inducing', 'add_cov_diag', 'standard_cross_entropy')
	values = (device, args.x_inducing_var, args.f_prior, args.n_inducing, args.add_cov_diag, args.standard_cross_entropy)
	fvi_args = dict(zip(keys, values))

	FVI = FVI_seg(x_size=(H_crop, W_crop), num_classes=num_classes, **fvi_args).to(device)
	optimizer = torch.optim.SGD(FVI.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

	if args.load:
		model_load_dir = os.path.join(args.base_dir, 'FVI_CV/model_{}_{}.bin'.format(args.dataset, exp_name))
		optimizer_load_dir = os.path.join(args.base_dir, 'FVI_CV/optimizer_{}_{}.bin'.format(args.dataset, exp_name)) 
		FVI.load_state_dict(torch.load(model_load_dir))
		optimizer.load_state_dict(torch.load(optimizer_load_dir))
		print('Loading FVI segmentation model..')

	if args.training_mode:
		print('Training FVI segmentation for {} epochs'.format(args.n_epochs))
		train(args.n_epochs, train_loader, FVI)
	if args.test_mode:
		print('Evaluating FVI segmentation on test set')
		model_load_dir = os.path.join(args.base_dir, 'FVI_CV/models_test/model_{}_fvi_seg_test.bin'.format(args.dataset))
		FVI.load_state_dict(torch.load(model_load_dir))
		error, mIOU = test(FVI, test_loader, num_classes, args.dataset, exp_name, mkdir=True)
		print('Test Error: {:.5f} || Test Mean IOU: {:.5f}'.format(error, mIOU))
		np.savetxt('{}_{}_epoch_{}_test_error.txt'.format(args.dataset, exp_name, -1), [error])
		np.savetxt('{}_{}_epoch_{}_test_mIOU.txt'.format(args.dataset, exp_name, -1), [mIOU])
	if args.test_runtime_mode:
		model_load_dir = os.path.join(args.base_dir, 'FVI_CV/models_test/model_{}_fvi_seg_test.bin'.format(args.dataset))
		FVI.load_state_dict(torch.load(model_load_dir))
		run_runtime_seg(FVI, test_loader, exp_name, 50)
