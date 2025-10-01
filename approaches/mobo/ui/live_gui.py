"""
Simple GUI for MOBO that shows the Pareto plot when optimization completes.
Since MOBO doesn't have continuous training like RL, we just show final results.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os
import threading
import time
from pathlib import Path


class MOBOResultsGUI:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.root = tk.Tk()
        self.root.title("MOBO PPA Optimization Results")
        self.root.geometry("900x700")

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Waiting for MOBO optimization to complete...")
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Results frame
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Optimization Results", padding="10")
        self.results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(1, weight=1)

        # Progress info
        self.info_text = tk.Text(self.results_frame, height=8, width=50)
        self.info_text.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))

        # Matplotlib figure for Pareto plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.results_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        self.refresh_btn = ttk.Button(button_frame, text="Refresh", command=self.refresh_results)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.export_btn = ttk.Button(button_frame, text="Export Plot", command=self.export_plot)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.close_btn = ttk.Button(button_frame, text="Close", command=self.root.quit)
        self.close_btn.pack(side=tk.LEFT)

        # Start monitoring for results
        self.monitor_results()

    def monitor_results(self):
        """Monitor the run directory for completion."""
        def check_completion():
            while True:
                if self.check_files_exist():
                    self.root.after(0, self.load_and_display_results)
                    break
                time.sleep(2)

        monitor_thread = threading.Thread(target=check_completion, daemon=True)
        monitor_thread.start()

    def check_files_exist(self):
        """Check if all expected output files exist."""
        required_files = ['candidates.csv', 'pareto.csv', 'progress.csv']
        return all((self.run_dir / f).exists() for f in required_files)

    def load_and_display_results(self):
        """Load results and update the display."""
        try:
            self.status_label.config(text="MOBO optimization completed! Loading results...")

            # Load data
            candidates_df = pd.read_csv(self.run_dir / 'candidates.csv')
            pareto_df = pd.read_csv(self.run_dir / 'pareto.csv')
            progress_df = pd.read_csv(self.run_dir / 'progress.csv')

            # Update info text
            self.update_info_text(candidates_df, pareto_df, progress_df)

            # Plot Pareto frontier
            self.plot_pareto_frontier(candidates_df, pareto_df)

            self.status_label.config(text="Results loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {str(e)}")
            self.status_label.config(text="Error loading results")

    def update_info_text(self, candidates_df, pareto_df, progress_df):
        """Update the information text widget."""
        info = []
        info.append("=== MOBO Optimization Summary ===\n")
        info.append(f"Total Evaluations: {len(candidates_df)}")
        info.append(f"Pareto Optimal Points: {len(pareto_df)}")
        info.append(f"Optimization Iterations: {len(progress_df)}")

        if len(progress_df) > 0:
            final_hv = progress_df['hv'].iloc[-1]
            info.append(f"Final Hypervolume: {final_hv:.6f}")

        info.append(f"\n=== Pareto Frontier Ranges ===")
        if len(pareto_df) > 0:
            power_range = (pareto_df['obj_0'].min(), pareto_df['obj_0'].max())
            delay_range = (pareto_df['obj_1'].min(), pareto_df['obj_1'].max())
            info.append(f"Power: [{power_range[0]:.4f}, {power_range[1]:.4f}]")
            info.append(f"Delay: [{delay_range[0]:.4f}, {delay_range[1]:.4f}]")

        info.append(f"\n=== Files Generated ===")
        info.append(f"• {self.run_dir}/candidates.csv")
        info.append(f"• {self.run_dir}/pareto.csv")
        info.append(f"• {self.run_dir}/progress.csv")
        info.append(f"• {self.run_dir}/pareto.png")
        info.append(f"• {self.run_dir}/attainment.png")

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(info))

    def plot_pareto_frontier(self, candidates_df, pareto_df):
        """Plot the Pareto frontier."""
        self.ax.clear()

        # Plot all candidates
        if len(candidates_df) > 0:
            feasible_mask = candidates_df.get('feasible', [True] * len(candidates_df))
            feasible_candidates = candidates_df[feasible_mask]

            if len(feasible_candidates) > 0:
                self.ax.scatter(feasible_candidates['obj_0'], feasible_candidates['obj_1'],
                              alpha=0.6, color='lightblue', s=30, label='Feasible Candidates')

        # Plot Pareto frontier
        if len(pareto_df) > 0:
            # Sort by first objective for clean line
            sorted_pareto = pareto_df.sort_values('obj_0')
            self.ax.plot(sorted_pareto['obj_0'], sorted_pareto['obj_1'],
                        'r-o', linewidth=2, markersize=6, label='Pareto Frontier')

        self.ax.set_xlabel('Power (minimize)')
        self.ax.set_ylabel('Delay (minimize)')
        self.ax.set_title('MOBO PPA Optimization: Pareto Frontier')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def refresh_results(self):
        """Manually refresh the results."""
        if self.check_files_exist():
            self.load_and_display_results()
        else:
            messagebox.showinfo("Info", "Results not ready yet. Please wait for optimization to complete.")

    def export_plot(self):
        """Export the current plot to a file."""
        try:
            export_path = self.run_dir / "pareto_gui_export.png"
            self.fig.savefig(export_path, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot exported to:\n{export_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export plot: {str(e)}")

    def show(self):
        """Show the GUI."""
        self.root.mainloop()


def show_mobo_gui(run_dir):
    """Launch the MOBO results GUI."""
    gui = MOBOResultsGUI(run_dir)
    gui.show()


if __name__ == "__main__":
    # Test with a sample run directory
    import sys
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = "../../runs/mobo_20250930_233945"  # Default for testing

    show_mobo_gui(run_dir)