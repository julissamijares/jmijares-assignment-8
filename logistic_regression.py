import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Function to generate ellipsoid clusters with a shift
def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) and apply shift along y = -x direction
    shift_vector = np.array([1, -1])
    shift_vector = shift_vector / np.linalg.norm(shift_vector)  # Normalize the vector
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    X2 += distance * shift_vector  # Apply shift along the y = -x direction
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


# Function to fit logistic regression and extract coefficients
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

# Main experiment function
def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list = [], [], []
    slope_list, intercept_list = [], []
    loss_list, margin_widths = [], []
    sample_data = {}

    n_samples = 8
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)

        # Fit logistic regression model
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)

        # Calculate slope and intercept
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Compute logistic loss
        prob = model.predict_proba(X)
        loss = log_loss(y, prob)
        loss_list.append(loss)

        # Plot the dataset and decision boundary
        ax = plt.subplot(n_rows, n_cols, i)
        ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0', alpha=0.6)
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1', alpha=0.6)

        x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

        # Calculate margin width between 70% confidence contours
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = ax.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = ax.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)

            if level == 0.7:
                class_1_vertices = class_1_contour.allsegs[0][0]
                class_0_vertices = class_0_contour.allsegs[0][0]
                distances = cdist(class_1_vertices, class_0_vertices)
                margin_width = np.min(distances)
                margin_widths.append(margin_width)

        ax.set_title(f"Shift Distance = {distance:.2f}", fontsize=16)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()

        # Adjust axis limits dynamically
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

        sample_data[distance] = (X, y, model, beta0, beta1, beta2)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plotting parameters vs shift distance
    plt.figure(figsize=(18, 15))

    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, 'o-')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, 'o-')
    plt.title("Shift Distance vs Beta1 (x1 Coefficient)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, 'o-')
    plt.title("Shift Distance vs Beta2 (x2 Coefficient)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, 'o-')
    plt.title("Shift Distance vs Slope (Beta1 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, 'o-')
    plt.title("Shift Distance vs Intercept (Beta0 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, 'o-')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, 'o-')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")


if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
