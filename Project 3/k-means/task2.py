import numpy as np
import cv2
import matplotlib.pyplot as plt

import time


class KMeans(object):


    def __init__(self, X, k, centers=None):
        self.X_ = np.array(X)
        self.k_ = k
        # a cluster of index of each point
        self.clusters_ = [[] for _ in range(k)]

        if centers is not None:
            assert(k == len(centers))
            self.centers_ = np.array([c for c in centers])
        else:
            self.centers_ = np.array([self.X_[i] for i in range(k)])

    
    def update_center(self):
        centers = []
        
        for cluster in self.clusters_:
            c = np.mean(self.X_[cluster], axis=0)
            centers.append(c)
        
        self.centers_ = np.array(centers)


    def classify(self):
        clusters = [[] for _ in range(self.k_)]
        for i, x in enumerate(self.X_):
            # squared difference between each center and x
            diff = (self.centers_ - x) ** 2
            # squared euclidean distance between each center and x
            dist = diff[:, 0] + diff[:, 1]
            # find closest center
            idx = np.argmin(dist)
            
            clusters[idx].append(i)
        
        self.clusters_ = clusters
    

    def run(self, n_iter=100):
        for i in range(n_iter):
            print("[INFO] Iteration: {}".format(i))
            prev_centers = self.centers_.copy()

            self.classify()
            self.update_center()

            diff = np.sum((self.centers_ - prev_centers) ** 2)
            
            # converaged
            if diff < 1e-4:
                break


    def get_labels(self):
        labels = {}

        for center_id, cluster in enumerate(self.clusters_):
            for idx in cluster:
                labels[idx] = center_id

        return labels

    
    def get_clusters(self):
        clusters = []

        for cluster in self.clusters_:
            clusters.append(self.X_[cluster])
        
        return clusters


    def get_centers(self):
        return np.array(self.centers_)


def plot_scatter(clusters, centers, output_dir):
    colors = [i for i in 'rgb']

    # K is number of clusters
    K = len(centers)

    txt_shift_x = -0.1
    txt_shift_y = -0.07
    # txt_shift_x = 0
    # txt_shift_y = 0

    cluster_x = [[] for x in range(K)]
    cluster_y = [[] for x in range(K)]

    for i in range(K):
        for item in clusters[i]:
            cluster_x[i].append(item[0])
            cluster_y[i].append(item[1])

    # plot data points
    for i in range(K):
        plt.scatter(cluster_x[i], cluster_y[i], s=40, c=colors[i % len(colors)], marker='^')
        
        for x, y in zip(cluster_x[i], cluster_y[i]):
            plt.annotate("({}, {})".format(x, y), (x + txt_shift_x, y + txt_shift_y))

    
    # plot centers
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], s=40, c=colors[i % len(colors)], marker='o')

        plt.annotate("({:.2f}, {:.2f})".format(c[0], c[1]), (c[0] + txt_shift_x, c[1] + txt_shift_y))

    plt.title(output_dir.split('.')[0])
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(output_dir, dpi=255)
    
    plt.clf()

def print_classification(clusters):
    print("-" * 60)
    for i, cluster in enumerate(clusters):
        for point in cluster:
            print("{}\t{}".format(point, i))


def print_center(centers):
    print("-" * 60)
    for c in centers:
        print("[{}, {}]".format(*c))


def reconstruct_img(img, pixel_labels, color_centers):
    recon_img = img.copy()

    h, w, d = img.shape

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            color = color_centers[pixel_labels[idx]]

            recon_img[i][j] = color

    recon_img = np.array(recon_img * 255, np.uint8)
    return recon_img


def classification():
    X = [
        [5.9, 3.2],
        [4.6, 2.9],
        [6.2, 2.8],
        [4.7, 3.2],
        [5.5, 4.2],
        [5.0, 3.0],
        [4.9, 3.1],
        [6.7, 3.1],
        [5.1, 3.8],
        [6.0, 3.0]
    ]

    kmeans = KMeans(X=X, k=3, centers=[(6.2, 3.2), (6.6, 3.7), (6.5, 3.0)])

    # iteration 1: classify
    kmeans.classify()
    clusters = kmeans.get_clusters()
    centers = kmeans.get_centers()

    print_classification(clusters)
    plot_scatter(clusters, centers, "task2_iter1_a.jpg")
    
    # iteration 1: update center
    kmeans.update_center()
    clusters = kmeans.get_clusters()
    centers = kmeans.get_centers()

    print_center(centers)
    plot_scatter(clusters, centers, "task2_iter1_b.jpg")

    # iteration 2: classify
    kmeans.classify()
    clusters = kmeans.get_clusters()
    centers = kmeans.get_centers()

    print_classification(clusters)
    plot_scatter(clusters, centers, "task2_iter2_a.jpg")

    # iteration 2: update center
    kmeans.update_center()
    clusters = kmeans.get_clusters()
    centers = kmeans.get_centers()

    print_center(centers)
    plot_scatter(clusters, centers, "task2_iter2_b.jpg")


def color_quantization():
    img = cv2.imread("baboon.png")

    # convert to float space, better performance
    img = np.array(img, dtype=np.float64) / 255.0
    h, w, d = img.shape

    for k in [3, 5, 10, 20]:
        img_array = np.reshape(img, (w * h, d))
        kmeans = KMeans(img_array, k=k)

        start_t = time.time()
        kmeans.run()
        end_t = time.time()

        print("[K = {}]: Time elasped: {} sec".format(k, end_t - start_t))

        pixel_labels = kmeans.get_labels()
        color_centers = kmeans.get_centers()

        quantized_img = reconstruct_img(img, pixel_labels, color_centers)

        cv2.imwrite("task2_baboon_{}.jpg".format(k), quantized_img)
        print("[K = {}]: Reconstructed image was saved at: ./task2_baboon_{}.jpg".format(k, k))


if __name__ == "__main__":

    classification()
    color_quantization()
