import cv2
import numpy as np

def filter2d(src, kernel, fill_value=-1):
	m, n = kernel.shape
	
	d = int((m-1)/2) # interval
	h, w = src.shape[0], src.shape[1]
	
	if fill_value == -1: dst = src.copy()
	elif fill_value == 0: dst = np.zeros((h, w))
	else:
			dst = np.zeros((h, w))
			dst.fill(fill_value)
	
	for y in range(d, h - d):
			for x in range(d, w - d):
					dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)
					
	return dst


# Non maximum Suppression
def non_max_sup(G, Gth):
	h, w = G.shape
	dst = G.copy()

	# 4 dirctinal gradients
	Gth[np.where((Gth >= -22.5) & (Gth < 22.5))] = 0
	Gth[np.where((Gth >= 157.5 ) & (Gth < 180))] = 0
	Gth[np.where((Gth >= -180 ) & (Gth < -157.5))] = 0
	Gth[np.where((Gth >= 22.5) & (Gth < 67.5))] = 45
	Gth[np.where((Gth >= -157.5 ) & (Gth < -112.5))] = 45
	Gth[np.where((Gth >= 67.5) & (Gth < 112.5))] = 90
	Gth[np.where((Gth >= -112.5) & (Gth < -67.5))] = 90
	Gth[np.where((Gth >= 112.5) & (Gth < 157.5))] = 135
	Gth[np.where((Gth >= -67.5) & (Gth < -22.5))] = 135

	for y in range(1, h - 1):
		for x in range(1, w - 1):
			if Gth[y][x]==0:
				if (G[y][x] < G[y][x+1]) or (G[y][x] < G[y][x-1]):
					dst[y][x] = 0
			elif Gth[y][x] == 45:
				if (G[y][x] < G[y-1][x+1]) or (G[y][x] < G[y+1][x-1]):
					dst[y][x] = 0
			elif Gth[y][x] == 90:
				if (G[y][x] < G[y+1][x]) or (G[y][x] < G[y-1][x]):
					dst[y][x] = 0
			else:
				if (G[y][x] < G[y+1][x+1]) or  (G[y][x] < G[y-1][x-1]):
					dst[y][x] = 0
	return dst


# Hysteresis Threshold
def hysteresis_threshold(src, t_min=75, t_max=150, d=1):

	h, w = src.shape
	dst = src.copy()

	for y in range(0, h):
		for x in range(0, w):
			
			if src[y][x] >= t_max: dst[y][x] = 255
			
			elif src[y][x] < t_min: dst[y][x] = 0
			
			else:
				if np.max(src[y-d:y+d+1, x-d:x+d+1]) >= t_max:
					dst[y][x] = 255
				else: dst[y][x] = 0

	return dst


def canny_edge_detecter(gray, t_min, t_max, d):

	# Gauusian
	kernel_g = np.array([[1/16, 1/8, 1/16],
												[1/8,  1/4,  1/8],
												[1/16, 1/8, 1/16]])

	G = filter2d(gray, kernel_g, -1)

	# Sobel
	kernel_sx = np.array([[-1,0,1],
												[-2,0,2],
												[-1,0,1]])
	kernel_sy =  np.array([[-1,-2,-1],
													[0,  0, 0],
													[1,  2, 1]])
	Gx = filter2d(G, kernel_sx, 0)
	Gy = filter2d(G, kernel_sy, 0)
	
	# Gradient Dirctions
	G = np.sqrt(Gx**2 + Gy**2)
	Gth = np.arctan2(Gy, Gx) * 180 / np.pi

	# Non maximum Suppression
	G = non_max_sup(G, Gth)

	# Hysteresis Threshold
	return hysteresis_threshold(G, t_min, t_max, d)


def main():
	# img = cv2.imread("data/HED-BSDS/train/aug_data/0.0_1_0/2092.jpg")
	img = cv2.imread("/home/gatheluck/Codes/hed/logs/vgg16/images/sample2T.png")

	gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
	#edge1 = canny_edge_detecter(gray, 100, 200, 1)
	edge1 = canny_edge_detecter(gray, 75, 150, 1)

	cv2.imwrite("output1.jpg", edge1)
    
if __name__ == "__main__":
  main()