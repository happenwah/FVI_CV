import numpy as np
import torch
import matplotlib.pyplot as plt

def run_runtime_seg(model, test_set, exp_name, S):
	X_test = list(test_set)[0][0].cuda()
	model.eval()
	#First forward pass, ignore
	time_list = []
	for _ in range(100 + 1):
		end = model.predict_runtime(X_test, S)
		time_list.append(end)
	time_list = np.array(time_list)[1:]
	time_mean = time_list.mean()
	time_std = time_list.std()
	np.savetxt('{}_mean_runtime_{}_samples.txt'.format(exp_name, S), [time_mean])
	np.savetxt('{}_std_runtime_{}_samples.txt'.format(exp_name, S), [time_std])
	print(time_list)
	print('Inference time elapsed (s), mean: {} || std: {}'.format(time_mean, time_std))
	model.train()

def calibration_per_image(probs, mask, y):
    mask_probs = mask[:,np.newaxis,:].repeat(probs.shape[1], 1)
    probs = probs.flatten()[mask_probs.flatten()]
    mask = mask.flatten()
    y = y.flatten()[mask]
    y = np.eye(11)[y].transpose().flatten()
    n_levels = 10
    true_freq = np.linspace(0., 1., n_levels)
    obs_freq = np.zeros_like(true_freq)
    pred_freq = np.zeros_like(true_freq)
    level_prev = 0.
    for i,level in enumerate(true_freq):
        mask_level = (probs > level_prev) & (probs <= level)
        if mask_level.astype(np.float32).sum() < 1.:
            pred_freq[i] = 0.
            obs_freq[i] = 0.
        else:
            pred_freq[i] = probs[mask_level].mean()
            obs_freq[i] = y[mask_level].mean()
        level_prev = level
    #Calibration score, uniform weighting bins
    calibration = ((obs_freq - pred_freq) ** 2 * 1.).sum()
    idx = np.argsort(pred_freq)
    return obs_freq[idx], pred_freq[idx], calibration

def plot_per_image(rgb, ground_truth, pred, pred_entropy, probs, mask, dataset, exp_name, idx, deterministic=False):
    if dataset == 'camvid':
        H, W = 360, 480
    im_ratio = float(H/W)
    rgb = rgb.view(3, H, W).permute(1, 2, 0).numpy()
    ground_truth = ground_truth.view(H, W).numpy()
    pred = pred.view(H, W).numpy()
    fig = plt.figure(1,figsize=(12, 2))
    ax1 = plt.subplot(151)
    im1 = ax1.imshow(rgb)
    ax1.axis('off')
    ax2 = plt.subplot(152)
    im2 = ax2.imshow(ground_truth)
    ax2.axis('off')
    ax3 = plt.subplot(153)
    im3 = ax3.imshow(pred)
    ax3.axis('off')
    if not deterministic:
        pred_entropy = pred_entropy.view(H, W).numpy()
        probs = probs.numpy()
        mask = mask.view(1, -1).numpy()
        ax4 = plt.subplot(154)
        im4 = ax4.imshow(pred_entropy, vmin=0., vmax=np.log(11.))
        ax4.axis('off')
        cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046*im_ratio, pad=0.04)
        cb4.ax.tick_params(labelsize=0)
        cb4.ax.tick_params(size=0)
        true_freq, pred_freq, calibration = calibration_per_image(probs, mask, ground_truth)
        print('Img: {} || Calibration: {:.5f}'.format(idx, calibration))
        ax5 = plt.subplot(155)                                                                                                                                          
        ax5.set_aspect(im_ratio)
        ax5.xaxis.set_tick_params(labelsize=5)
        ax5.yaxis.set_tick_params(labelsize=5)
        ax5.plot(pred_freq, true_freq, color='red')
        ax5.plot([0., 1.], [0., 1.], 'g--')
        np.savetxt('{}_{}_calibration_score_{}.txt'.format(dataset, exp_name, idx), [calibration])
    plt.savefig('{}_{}_results_test_pred_{}.pdf'.format(dataset, exp_name, idx), bbox_inches='tight', pad_inches=0.1)
    plt.close()

def test(
    model,
    test_loader,
    num_classes,
    dataset,
    exp_name,
    plot_imgs=True,
    mkdir=False
         ):
    model.eval()
    H, W = 360, 480
    if mkdir:
        import os
        new_dir = './results_{}'.format(exp_name)
        os.makedirs(new_dir, exist_ok=True)
        os.chdir(new_dir)
        n_save = len(test_loader)
    else:
        n_save = 15
    with torch.no_grad():
        test_loss = 0
        test_error = 0
        I_tot = np.zeros(num_classes)
        U_tot = np.zeros(num_classes)

        for idx, (data, target) in enumerate(list(test_loader)):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            print('Processing img {}'.format(idx))
            if idx < n_save and plot_imgs:
                output, entropy, probs = model.predict(data, return_probs=True)
                mask_ = 11. * torch.ones_like(target)
                mask = torch.ne(target, mask_)
                output = output.view(1, H, W)
                output[~mask] = 11.
                plot_per_image(data.cpu(), target.cpu(), output.cpu(), entropy.cpu(), probs.cpu(), mask.cpu(), dataset, exp_name, idx)
            else:
                output = model.predict(data)

            I, U, acc = numpy_metrics(
                output.view(target.size(0), -1).cpu().numpy(),
                target.view(target.size(0), -1).cpu().numpy(),
                n_classes=11,
                void_labels=[11],
            )
            I_tot += I
            U_tot += U
            test_error += 1 - acc


        test_error /= len(test_loader)
        m_jacc = np.mean(I_tot / U_tot)

        return test_error, m_jacc


def numpy_metrics(y_pred, y_true, n_classes=11, void_labels=[11]):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    from: https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    void label is 11 by default
    """

    # Put y_pred and y_true under the same shape
    assert y_pred.shape == y_true.shape, "shapes do not match"

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)
    return I, U, accuracy
