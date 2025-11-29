import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
import mplcursors

from src.color_conversion import *

# Displays an individual's predicted and actual favorite color in RGB and HSV values
def show_test_details(predicted_hsv, actual_hsv, test_names):
    while True:
        try:
            test_index = int(input(f"Enter an index of the test set (1-{len(test_names)}, -1 to exit): "))
        except ValueError: # Occurs when the user enters a non-integer
            print("Invalid input, please enter an integer.")
            continue
        if test_index == -1: # Allows user to go back to menu
            return
        elif 1 <= test_index <= len(test_names): # Displays information of an individual
            test_index -= 1 # Converts the index back to 0 starting index
            name = test_names[test_index] # Gets the name of the individual

            # HSV values already denormalized
            pred_hsv = predicted_hsv[test_index]
            act_hsv = actual_hsv[test_index]

            # Convert HSV → RGB
            pred_rgb = hsv_to_rgb(pred_hsv)
            act_rgb = hsv_to_rgb(act_hsv)

            # --- Print details ---
            print(f"\nName: {name}")
            print(f"Predicted RGB: ({pred_rgb[0]}, {pred_rgb[1]}, {pred_rgb[2]})")
            print(f"Actual RGB:    ({act_rgb[0]}, {act_rgb[1]}, {act_rgb[2]})")
            print(f"Predicted HSV: ({pred_hsv[0]:.0f}°, {pred_hsv[1]:.0f}%, {pred_hsv[2]:.0f}%)")
            print(f"Actual HSV:    ({act_hsv[0]:.0f}°, {act_hsv[1]:.0f}%, {act_hsv[2]:.0f}%)\n")

            # Visual Comparison
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))

            # Normalizes RGB for matplotlib (0–1)
            pred_rgb_norm = [[[c / 255 for c in pred_rgb]]]
            act_rgb_norm = [[[c / 255 for c in act_rgb]]]

            axes[0].imshow(pred_rgb_norm)
            axes[0].set_title("Predicted Color")
            axes[0].axis("off")

            axes[1].imshow(act_rgb_norm)
            axes[1].set_title("Actual Color")
            axes[1].axis("off")

            fig.suptitle(f"{name}'s Favorite Color Comparison")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Invalid index, try again.")

# Displays a polar graph showing the difference between the predict hue vs. the actual hue in the test set
def plot_hue_predictions(y_pred, y_actual, test_names):
    # Formats the polar graph
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={'projection': 'polar'})
    plt.title("Test Set Hue Predictions vs. Actual", fontsize=18, pad=30, loc='center')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    degree_angles = np.arange(0, 360, 30)
    ax.set_xticks(np.radians(degree_angles))
    ax.set_xticklabels([str(d) for d in degree_angles])
    for angle in np.radians(degree_angles):
        ax.plot([angle, angle], [1.02, 1.05], color="black", lw=1, zorder=10)
    for angle in np.radians(degree_angles):
        ax.plot([angle, angle], [0, 1.0], color="gray", lw=1, ls="--", alpha=0.6, zorder=1)
    ax.grid(False)
    # Creates the color wheel
    theta = np.linspace(0, 2 * np.pi, 360)
    for t in theta:
        color = np.array(hsv_to_rgb([np.degrees(t) % 360, 100, 100])) / 255.0
        ax.bar(t, 1.0, width=2 * np.pi / 360, bottom=0, color=color, edgecolor='none', zorder=0)

    # Sets up the radius
    radius_actual, radius_pred = 1.0, 0.5
    theta_full = np.linspace(0, 2 * np.pi, 400)
    ax.plot(theta_full, [radius_pred] * len(theta_full),
            color='gray', lw=1, ls='--', alpha=0.6, zorder=1)

    # Gets hues
    hue_actual = y_actual[:, 0] % 360
    hue_pred = y_pred[:, 0] % 360
    # Rounds to whole degrees
    hue_actual_rounded = np.round(hue_actual).astype(int)
    hue_pred_rounded = np.round(hue_pred).astype(int)

    # Gets the colors
    colors_actual = [np.array(hsv_to_rgb([h, 100, 100])) / 255.0 for h in hue_actual_rounded]
    colors_pred = [np.array(hsv_to_rgb([h, 100, 100])) / 255.0 for h in hue_pred_rounded]

    # Plots actual circles
    scatter_actual = ax.scatter(np.radians(hue_actual_rounded),
                                np.ones_like(hue_actual_rounded) * radius_actual,
                                s=150, c=colors_actual, edgecolors='k', zorder=4)

    # Triangle marker base
    verts = np.array([[0.0, 1.0], [-0.6, -1.0], [0.6, -1.0], [0.0, 1.0]])
    triangle_marker = Path(verts)

    # Collects predicted points for one scatter
    scatter_pred_x = []
    scatter_pred_y = []
    pred_markers = []
    pred_colors = []
    for h_actual, h_pred, color in zip(hue_actual_rounded, hue_pred_rounded, colors_pred):
        h_actual_rad = math.radians(h_actual)
        h_pred_rad = math.radians(h_pred)
        # Cartesian coordinates
        actual_x = radius_actual * math.sin(h_actual_rad)
        actual_y = radius_actual * math.cos(h_actual_rad)
        pred_x = radius_pred * math.sin(h_pred_rad)
        pred_y = radius_pred * math.cos(h_pred_rad)
        # Gets orientation of triangle
        vector_x, vector_y = actual_x - pred_x, actual_y - pred_y
        orientation = math.degrees(math.atan2(vector_y, vector_x))
        # Saves the Cartesian coordinates, triangle orientation, and color
        scatter_pred_x.append(h_pred_rad)
        scatter_pred_y.append(radius_pred)
        pred_markers.append(MarkerStyle(triangle_marker, transform=Affine2D().rotate_deg(orientation - 90)))
        pred_colors.append(color)
        # Line connecting pred to actual
        diff = (h_actual - h_pred + 180) % 360 - 180
        mid_hue = (h_pred + diff / 2) % 360
        line_color = np.array(hsv_to_rgb([mid_hue, 100, 75])) / 255.0
        ax.plot([h_pred_rad, h_actual_rad], [radius_pred, radius_actual],
                color=line_color, lw=1.5, zorder=3)

    # Creates one scatter for all predicted triangles
    scatter_pred = ax.scatter(scatter_pred_x, scatter_pred_y, s=200,
                              c=pred_colors, edgecolors='k', zorder=5)

    # Applies per-point rotated triangle shapes
    scatter_pred.set_paths([m.get_path().transformed(m.get_transform()) for m in pred_markers])

    ax.set_ylim(0, 1.05)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=12, label='Actual'),
        Line2D([0], [0], marker=triangle_marker, color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=12, label='Predicted')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.15))

    if mplcursors is not None:
        cursor = mplcursors.cursor([scatter_actual, scatter_pred], hover=True, multiple=False)

        @cursor.connect("add")
        def on_add(sel, H=hue_actual_rounded, P=hue_pred_rounded, N=test_names):
            i = sel.index
            diff = (P[i] - H[i] + 180) % 360 - 180
            if diff in (-180, 180):
                diff_str = "±180"
            elif diff > 0:
                diff_str = f"+{diff}"
            elif diff < 0:
                diff_str = f"{diff}"
            else:
                diff_str = "0"

            sel.annotation.set_text(
                f"Name: {N[i]}"
                f"\nPred: {P[i]}°"
                f"\nActual: {H[i]}°"
                f"\nError: {diff_str}°"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.85)

    plt.tight_layout()
    plt.show()

# Displays 2 scatter plots showing the difference between the predicted vs. the actual saturation and value in the test set
def plot_sat_val_predictions(predicted_hsv, actual_hsv, test_names):
    hsv_actual_fixed = np.array(actual_hsv)
    hsv_pred_fixed = np.array(predicted_hsv)
    # Gets saturation and value (0–100 range)
    s_actual, v_actual = hsv_actual_fixed[:, 1], hsv_actual_fixed[:, 2]
    s_pred, v_pred = hsv_pred_fixed[:, 1], hsv_pred_fixed[:, 2]

    # Create side-by-side scatter plots (single figure)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    datasets = [
        ("Saturation: Actual vs. Predicted", s_actual, s_pred),
        ("Value: Actual vs. Predicted", v_actual, v_pred),
    ]

    for ax, (title, actual, pred) in zip(axes, datasets):
        # Round values for plotting
        actual_rounded = np.round(actual).astype(int)
        pred_rounded = np.round(pred).astype(int)

        # Diagonal line (above error lines but below points)
        ax.plot([0, 100], [0, 100], color="lightgray", linestyle="--", lw=2, zorder=2)

        # Scatter points (on top)
        scatter = ax.scatter(actual_rounded, pred_rounded, color="dimgray", edgecolors="k",
                             marker="o", zorder=3)

        # Vertical error lines (underneath)
        for x, y in zip(actual_rounded, pred_rounded):
            if y > x: # overestimate
                ax.plot([x, x], [x, y], color="deepskyblue", lw=1.2, zorder=1, alpha=0.7)
            elif y < x: # underestimate
                ax.plot([x, x], [y, x], color="darkorange", lw=1.2, zorder=1, alpha=0.7)
            # No line drawn if y == x

        # Labels, limits, grid
        ax.set_title(title)
        ax.set_xlabel("Actual (%)")
        ax.set_ylabel("Predicted (%)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xticks(np.arange(0, 101, 10))
        ax.set_yticks(np.arange(0, 101, 10))
        ax.grid(True, which="both", color="lightgray", linestyle="--", linewidth=0.7)

        # Hover tooltips (name + Pred above Actual + Error)
        if mplcursors is not None:
            cursor = mplcursors.cursor(scatter, hover=True)
            @cursor.connect("add")
            def on_add(sel, A=actual_rounded, P=pred_rounded, N=test_names):
                i = sel.index
                diff = P[i] - A[i]
                if diff > 0:
                    diff_str = f"+{diff}"
                elif diff < 0:
                    diff_str = f"{diff}"
                else:
                    diff_str = "0"

                sel.annotation.set_text(
                    f"Name: {N[i]}"
                    f"\nPred: {P[i]}%"
                    f"\nActual: {A[i]}%"
                    f"\nError: {diff_str}%"
                )
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.85)

    plt.suptitle("Predicted vs. Actual Saturation and Value", fontsize=16)
    plt.tight_layout()
    plt.show()