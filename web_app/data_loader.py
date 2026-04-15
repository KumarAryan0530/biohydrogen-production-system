"""
Data loader for web dashboard.
Reads CSV files and image data from results directory.
"""

import os
import json
import pandas as pd
import base64
from pathlib import Path


class DataLoader:
    """Load and serve data from results directory."""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        
    def load_summary(self):
        """Load summary data from summary.csv"""
        csv_path = os.path.join(self.results_dir, 'summary.csv')
        if not os.path.exists(csv_path):
            return {}
        
        df = pd.read_csv(csv_path)
        summary = {}
        for _, row in df.iterrows():
            summary[row['Parameter']] = row['Value']
        return summary
    
    def load_sweep_results(self):
        """Load parameter sweep results."""
        csv_path = os.path.join(self.results_dir, 'sweep_results.csv')
        if not os.path.exists(csv_path):
            return None
        return pd.read_csv(csv_path)
    
    def load_baseline_simulation(self):
        """Load baseline simulation results."""
        csv_path = os.path.join(self.results_dir, 'baseline_simulation.csv')
        if not os.path.exists(csv_path):
            return None
        return pd.read_csv(csv_path)
    
    def load_mpc_feed_rates(self):
        """Load MPC feed rate schedule."""
        csv_path = os.path.join(self.results_dir, 'mpc_feed_rates.csv')
        if not os.path.exists(csv_path):
            return None
        return pd.read_csv(csv_path)
    
    def get_image_base64(self, filename):
        """Load image and convert to base64 for embedding in HTML."""
        img_path = os.path.join(self.results_dir, filename)
        if not os.path.exists(img_path):
            return None
        
        with open(img_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode()
    
    def get_dashboard_data(self):
        """Compile all data for dashboard."""
        summary = self.load_summary()
        
        data = {
            'summary': summary,
            'sweep_results': self.load_sweep_results().to_dict() if self.load_sweep_results() is not None else None,
            'baseline_simulation': self.load_baseline_simulation().to_dict() if self.load_baseline_simulation() is not None else None,
            'mpc_feed_rates': self.load_mpc_feed_rates().to_dict() if self.load_mpc_feed_rates() is not None else None,
            'heatmap': self.get_image_base64('h2_yield_heatmap.png'),
            'baseline_plot': self.get_image_base64('baseline_h2_production.png'),
            'cost_breakdown': self.get_image_base64('cost_breakdown.png'),
        }
        return data
