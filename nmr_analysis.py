"""
2D NMR Data Analysis Script
Simplified from: cell_data_V_20260626_800M_MCF.ipynb
"""

import os
import datetime
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.spatial import KDTree
from scipy import stats


def printbar():
    """Print timestamp"""
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 8 + nowtime)


def load_nmr_data(exp_path, expnos, data_type="hsqc"):
    """
    Load NMR data from Bruker format

    Parameters:
    -----------
    exp_path : str
        Path to experiment directory
    expnos : list
        List of experiment numbers
    data_type : str
        'hsqc' for even expnos, 'noesy' for odd expnos

    Returns:
    --------
    list : List of NMR data arrays
    """
    data_list = []
    for expno in expnos:
        file_path = f"{exp_path}/{expno}/pdata/1"
        try:
            dic, data = ng.bruker.read_pdata(file_path)
            data_list.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    return data_list


def detect_peaks(image, threshold=0.05, sigma=0.5, window_size=5, min_distance=5):
    """
    Detect peaks in 2D NMR data

    Parameters:
    -----------
    image : numpy.ndarray
        2D NMR data
    threshold : float
        Intensity threshold for peak detection
    sigma : float
        Gaussian smoothing parameter
    window_size : int
        Local maximum filter window size
    min_distance : int
        Minimum distance between peaks (pixels)

    Returns:
    --------
    tuple : (y_coords, x_coords) of detected peaks
    """
    # Smooth the image to reduce noise
    img_smooth = gaussian_filter(image, sigma=sigma)

    # Find local maxima
    local_max = maximum_filter(img_smooth, size=window_size) == img_smooth

    # Apply threshold
    peaks_mask = local_max & (img_smooth > threshold)

    # Get candidate peak positions
    y_candidates, x_candidates = np.where(peaks_mask)

    if len(x_candidates) == 0:
        return np.array([]), np.array([])

    # Sort by intensity (descending)
    coords = np.column_stack((x_candidates, y_candidates))
    intensities = img_smooth[y_candidates, x_candidates]
    order = intensities.argsort()[::-1]
    coords = coords[order]

    # Filter peaks by minimum distance using KDTree
    keep = np.ones(len(coords), dtype=bool)
    tree = KDTree(coords)

    for i in range(len(coords)):
        if keep[i]:
            neighbors = tree.query_ball_point(coords[i], r=min_distance)
            for j in neighbors:
                if j > i:
                    keep[j] = False

    return coords[keep, 1], coords[keep, 0]  # y, x


def calculate_peak_intensities(hsqc_array, x_arr, y_arr, window=2):
    """
    Calculate peak intensities across multiple HSQC spectra

    Parameters:
    -----------
    hsqc_array : numpy.ndarray
        3D array of HSQC spectra (n_spectra, y_dim, x_dim)
    x_arr : numpy.ndarray
        X coordinates of peaks
    y_arr : numpy.ndarray
        Y coordinates of peaks
    window : int
        Window size for local maximum

    Returns:
    --------
    list : Peak intensities for each peak across all spectra
    """
    all_peak_int = []
    for h, c in zip(x_arr, y_arr):
        peak_int = []
        for hsqc in hsqc_array:
            peak_int.append(
                hsqc[c - window : c + window, h - window : h + window].max()
            )
        all_peak_int.append(peak_int)
    return all_peak_int


def find_significant_peaks(data, percentile=90):
    """
    Find significantly changing peaks based on standard deviation

    Parameters:
    --------
    data : numpy.ndarray
        Peak intensity data (n_peaks, n_spectra)
    percentile : float
        Percentile threshold for significance

    Returns:
    --------
    tuple : (significant_mask, indices)
    """
    std_vals = data.std(axis=1)
    std_threshold = np.percentile(std_vals, percentile)
    significant_peaks = std_vals > std_threshold
    return significant_peaks, np.where(significant_peaks)[0]


def plot_hsqc_spectrum(
    hsqc_data,
    h1_sw,
    c13_sw,
    peaks_x=None,
    peaks_y=None,
    title="HSQC Spectrum",
    figsize=(10, 4),
):
    """Plot HSQC spectrum with optional peak markers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    edlev = np.linspace(0.05, 1, 100)

    # Plot without chemical shift labels
    ax1.contour(hsqc_data, edlev)
    if peaks_x is not None and peaks_y is not None:
        ax1.scatter(peaks_x, peaks_y, color="red", s=2)
    ax1.set_title("Index-based")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot with chemical shift labels
    ax2.contour(h1_sw, c13_sw, hsqc_data, edlev)
    if peaks_x is not None and peaks_y is not None:
        ax2.scatter(h1_sw[peaks_x], c13_sw[peaks_y], color="red", s=2)
    ax2.set_xlim(h1_sw[0], h1_sw[-1])
    ax2.set_ylim(c13_sw[0], c13_sw[-1])
    ax2.set_title(title)
    ax2.set_xlabel("1H (ppm)")
    ax2.set_ylabel("13C (ppm)")

    plt.tight_layout()
    plt.show()


def plot_noesy_spectrum(
    noesy_np, noesy_sw, peaks_x=None, xlim=None, ylim=None, title="NOESY Spectrum"
):
    """Plot NOESY spectrum"""
    plt.figure(figsize=(12, 6))
    plt.plot(noesy_sw, noesy_np.T, alpha=0.5)

    if peaks_x is not None:
        plt.plot(noesy_sw[peaks_x * 16], noesy_np[0][peaks_x * 16], "ro", markersize=6)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel("1H (ppm)")
    plt.ylabel("Intensity")
    plt.show()


def plot_peak_intensity_changes(
    data, significant_peaks, significant_labels=None, title="Peak Intensity Changes"
):
    """Plot peak intensity changes over time"""
    x0 = np.arange(data.shape[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Left plot: all peaks
    for row in data:
        ax1.plot(x0, row, color="#e0e0e0", alpha=0.2, linewidth=0.8)

    colors_light = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"]
    for i, row in enumerate(data[significant_peaks]):
        ax1.plot(
            x0, row, color=colors_light[i % len(colors_light)], alpha=0.6, linewidth=1.2
        )

    ax1.set_title(f"All {len(data)} Peaks", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time / Condition Index", fontsize=12)
    ax1.set_ylabel("Intensity", fontsize=12)
    ax1.grid(alpha=0.3, linestyle="--")

    # Right plot: significant peaks
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, row in enumerate(data[significant_peaks]):
        label = (
            f"{significant_labels[i]}"
            if significant_labels is not None
            else f"Peak {i}"
        )
        ax2.plot(
            x0,
            row,
            color=colors[i % len(colors)],
            alpha=1.0,
            linewidth=2.5,
            label=label,
        )

    ax2.set_title(
        f"Significant Peaks (n={np.sum(significant_peaks)})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Time / Condition Index", fontsize=12)
    ax2.set_ylabel("Intensity", fontsize=12)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Analysis
# ============================================================================
if __name__ == "__main__":
    printbar()

    # Configuration
    BASE_PATH = "./data/"

    # Get sorted experiment list
    files = os.listdir(BASE_PATH)
    files_sorted = sorted(files, key=lambda x: x[:8])

    # Select experiment (MCF cells)
    exp_idx = 2  # MCF
    exp = files_sorted[exp_idx]
    exp_path = f"{BASE_PATH}{exp}"

    print(f"Experiment: {exp}")

    # Load HSQC data (odd expnos: 3, 5, 7, ...)
    hsqc_expnos = list(range(3, 32, 2))
    hsqc_list = load_nmr_data(exp_path, hsqc_expnos, "hsqc")
    print(f"Loaded {len(hsqc_list)} HSQC spectra")

    # Load NOESY data (even expnos: 4, 6, 8, ...)
    noesy_expnos = list(range(4, 33, 2))
    noesy_list = load_nmr_data(exp_path, noesy_expnos, "noesy")
    print(f"Loaded {len(noesy_list)} NOESY spectra")

    printbar()

    # Normalize HSQC data
    hsqc_array = np.array(hsqc_list) / hsqc_list[0].max()
    print(f"HSQC array shape: {hsqc_array.shape}")

    # Define chemical shift axes
    h1_sw = np.linspace(4.691 + 16.0216 / 2, 4.691 - 16.0216 / 2, 4 * 1024)
    c13_sw = np.linspace(40 + 79.9988 / 2, 40 - 79.9988 / 2, 128)

    # Detect peaks in first HSQC spectrum
    hsqc_2d = hsqc_array[0]
    y_peaks, x_peaks = detect_peaks(
        hsqc_2d, threshold=0.05, sigma=0.5, window_size=5, min_distance=5
    )

    print(f"\nDetected {len(x_peaks)} peaks")

    # Plot HSQC spectrum with detected peaks
    plot_hsqc_spectrum(hsqc_2d, h1_sw, c13_sw, x_peaks, y_peaks, title=f"HSQC - {exp}")

    printbar()

    # Calculate peak intensities across all spectra
    all_peak_int = calculate_peak_intensities(hsqc_array, x_peaks, y_peaks)

    # Find significantly changing peaks
    data = np.array(all_peak_int)
    significant_peaks, sig_indices = find_significant_peaks(data, percentile=90)

    print(f"\nTotal peaks: {len(data)}")
    print(f"Significantly changing peaks: {np.sum(significant_peaks)}")
    print(f"Indices: {sig_indices}")

    # Plot peak intensity changes
    plot_peak_intensity_changes(data, significant_peaks, y_peaks[significant_peaks])

    printbar()
    print("Analysis complete!")
