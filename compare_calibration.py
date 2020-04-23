import glob
import numpy as np

def compare_calibration_seg():
	mcd_files = glob.glob('results_camvid_segmentation_mcd/*_calibration_*.txt')
	fvi_files = glob.glob('results_camvid_segmentation_gp_bnn/*_calibration_*.txt')
	mcd_files.sort()
	fvi_files.sort()
	assert len(mcd_files) == len(fvi_files)
	N = len(mcd_files)
	cal_mcd_mean = 0.
	cal_fvi_mean = 0.
	for i in range(N):
		cal_mcd = np.loadtxt(mcd_files[i])
		cal_fvi = np.loadtxt(fvi_files[i])
		cal_mcd_mean += cal_mcd
		cal_fvi_mean += cal_fvi
	cal_mcd_mean /= N
	cal_fvi_mean /= N
	print('MCD segmentation mean calibration score: {:.5f}'.format(cal_mcd_mean))
	print('FVI segmentation mean calibration score: {:.5f}'.format(cal_fvi_mean))

def compare_calibration_depth():
	fvi_gauss_files = glob.glob('results_make3d_fvi_gaussian_gp_bnn/*_calibration.txt')
	fvi_laplace_files = glob.glob('results_make3d_fvi_laplace_gp_bnn/*_calibration.txt')
	fvi_berhu_files = glob.glob('results_make3d_fvi_berhu_gp_bnn/*_calibration.txt')
	mcd_files = glob.glob('results_make3d_mc_dropout_p=0.2/*_calibration.txt')
	fvi_gauss_files.sort()
	fvi_laplace_files.sort()
	fvi_berhu_files.sort()
	mcd_files.sort()
	assert len(fvi_gauss_files) == len(fvi_laplace_files) \
			== len(fvi_berhu_files) == len(mcd_files)
	N = len(fvi_gauss_files)
	cal_gauss_mean = 0.
	cal_laplace_mean = 0.
	cal_berhu_mean = 0.
	cal_mcd_mean = 0.
	for i in range(N):
		cal_gauss = np.loadtxt(fvi_gauss_files[i])
		cal_laplace = np.loadtxt(fvi_laplace_files[i])
		cal_berhu = np.loadtxt(fvi_berhu_files[i])
		cal_mcd = np.loadtxt(mcd_files[i])
		cal_gauss_mean += cal_gauss
		cal_laplace_mean += cal_laplace
		cal_berhu_mean += cal_berhu
		cal_mcd_mean += cal_mcd
	cal_gauss_mean /= N
	cal_laplace_mean /= N
	cal_berhu_mean /= N
	cal_mcd_mean /= N
	print('FVI gaussian mean calibration score: {:.5f}'.format(cal_gauss_mean))
	print('FVI laplace mean calibration score: {:.5f}'.format(cal_laplace_mean))
	print('FVI berhu mean calibration score: {:.5f}'.format(cal_berhu_mean))
	print('MCD mean calibration score: {:.5f}'.format(cal_mcd_mean))

if __name__ == '__main__':
	compare_calibration_depth()
	compare_calibration_seg()
