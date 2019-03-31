import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KMean:
    ## Class Variables ##
    X = n_k = _tolerance = _max_itr = cluster = None
    centroids = []

    ## Constructors ##
    def __init__(self, n_k, tolerance=0.0001, max_itr=1000):
        self.n_k = n_k
        self._tolerance = tolerance
        self._max_itr = max_itr

    ## Methods ##
    def parse_data(self, file_path, sep):
        df = pd.read_csv(file_path, sep=sep)
        self.X = df.values

    def clusterify(self):
        for i in range(self._max_itr):
            self._init_cluster_classes()
            self._classify_points_to_cluster()

            new_centroid = self._update_centroid()

            if self._has_converged(new_centroid):
                self.centroids = new_centroid
                break
            else:
                self.centroids = new_centroid

    def _classify_points_to_cluster(self):
        if len(self.centroids) == 0:  # RANDOM init centroid
            for point in self.X:
                self.cluster[np.random.randint(0, len(self.cluster))].append(point)
        else:
            for point in self.X:
                dist = [euclidean_dist(point, centroid) for centroid in self.centroids]
                classified_index = dist.index(min(dist))
                self.cluster[classified_index].append(point)

    def _update_centroid(self):
        new_centroid = []

        for i, cluster in enumerate(self.cluster):
            new_centroid.append(np.average(cluster, axis=0))

        return new_centroid

    def _has_converged(self, new_centroid):
        if len(self.centroids) == 0:
            return False
        else:
            for i, centroid in enumerate(self.centroids):
                old_centroid = centroid
                curr_centroid = new_centroid[i]

                if np.sum((curr_centroid - old_centroid) / old_centroid * 100.0) > self._tolerance:
                    return False
                else:
                    return True

    def _init_cluster_classes(self):
        self.cluster = []

        for i in range(self.n_k):
            self.cluster.append([])

    def init_cluster_center(self, method="FORGY"):
        if method == "FIRST_K":
            for i in range(self.n_k):
                self.centroids.append(self.X[i])
        elif method == "FORGY":
            index_list = np.random.choice(len(self.X), self.n_k)
            for index in index_list:
                self.centroids.append(self.X[index])
        elif method == "RANDOM":
            return
        else:
            raise NotImplementedError

    @staticmethod
    def euclidean_dist(p1, p2):
        sqr_dist = 0
        for i in range(len(p1)):
            sqr_dist += (p1[i] - p2[i]) ** 2

        return np.sqrt(sqr_dist)


def main():
    k_mean = KMean(2)
    k_mean.parse_data("heightWeight.csv", ',')
    k_mean.init_cluster_center("FORGY")
    k_mean.clusterify()

    colors = 10 * ['r', 'g', 'c', 'b', 'k']

    for i, cluster in enumerate(k_mean.cluster):
        color = colors[i]
        plt.scatter(k_mean.centroids[i][0], k_mean.centroids[i][1], s=130, color=color, marker='x')

        for points in cluster:
            plt.scatter(points[0], points[1], color=color, s=30)

    plt.show()


if __name__ == "__main__":
    main()
