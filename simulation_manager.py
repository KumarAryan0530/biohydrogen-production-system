"""
Simulation Manager - Handles parameter storage, versioning, and result management.
Allows non-technical users to run simulations with custom parameters and track history.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil


class SimulationManager:
    """Manages simulation parameters, runs, and versioned results."""
    
    def __init__(self, project_root='c:/Users/aryan/Desktop/Biohydrogen'):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'results'
        self.runs_dir = self.project_root / 'results_runs'
        self.metadata_file = self.runs_dir / 'runs_metadata.json'
        
        # Create directories if they don't exist
        self.runs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load runs metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'runs': []}
    
    def _save_metadata(self):
        """Save runs metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_default_parameters(self):
        """Return default simulation parameters."""
        return {
            'run_name': 'Fast Default Run',
            'baseline_ph': 5.5,
            'baseline_temp': 35.0,
            'baseline_days': 10.0,
            'sweep_ph_min': 5.0,
            'sweep_ph_max': 8.0,
            'sweep_ph_step': 1.0,
            'sweep_temp_min': 25,
            'sweep_temp_max': 40,
            'sweep_temp_step': 5,
            'sweep_days': 10.0,
            'mpc_control_horizon': 5.0,
            'mpc_control_interval': 5.0,
            'mpc_total_time': 10.0,
            'plant_capital_cost': 5000000,
            'plant_life': 20,
            'discount_rate': 0.08,
            'capacity_factor': 0.90,
        }
    
    def create_run(self, parameters):
        """
        Create a new simulation run with given parameters.
        
        Parameters
        ----------
        parameters : dict
            Simulation parameters
            
        Returns
        -------
        run_id : str
            Unique identifier for this run
        run_dir : Path
            Directory where results will be stored
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = parameters.get('run_name', 'Run')
        run_id = f"{timestamp}_{run_name.replace(' ', '_')}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Store parameters
        params_file = run_dir / 'parameters.json'
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        return run_id, run_dir
    
    def save_run_results(self, run_id, run_dir, results_dict):
        """
        Save simulation results for a run.
        
        Parameters
        ----------
        run_id : str
            Run identifier
        run_dir : Path
            Run directory
        results_dict : dict
            Dictionary with results from simulation (model, summary, etc.)
        """
        # Load parameters for this run
        params_file = run_dir / 'parameters.json'
        with open(params_file, 'r') as f:
            parameters = json.load(f)
        
        # Create summary
        summary = {
            'run_id': run_id,
            'run_name': parameters.get('run_name', 'Run'),
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'results': results_dict,
        }
        
        # Save summary as JSON
        summary_file = run_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Add to metadata
        metadata_entry = {
            'run_id': run_id,
            'run_name': parameters.get('run_name', 'Run'),
            'timestamp': datetime.now().isoformat(),
            'optimal_ph': float(results_dict.get('optimal_ph', 0)),
            'optimal_temp': float(results_dict.get('optimal_temp', 0)),
            'optimal_h2_yield': float(results_dict.get('optimal_h2_yield', 0)),
            'mpc_improvement': float(results_dict.get('mpc_improvement', 0)),
            'lcoh': float(results_dict.get('lcoh', 0)),
            'annual_h2_kg': float(results_dict.get('annual_h2_kg', 0)),
        }
        
        self.metadata['runs'].append(metadata_entry)
        self._save_metadata()
        
        # Copy CSV and image files to run directory
        for file_pattern in ['*.csv', '*.png']:
            for file in self.results_dir.glob(file_pattern):
                shutil.copy(file, run_dir / file.name)
    
    def get_all_runs(self):
        """Get list of all simulation runs."""
        return sorted(
            self.metadata['runs'],
            key=lambda x: x['timestamp'],
            reverse=True  # Most recent first
        )
    
    def get_run_details(self, run_id):
        """Get detailed information about a specific run."""
        run_dir = self.runs_dir / run_id
        
        if not run_dir.exists():
            return None
        
        # Load summary
        summary_file = run_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Load parameters
        params_file = run_dir / 'parameters.json'
        if params_file.exists():
            with open(params_file, 'r') as f:
                parameters = json.load(f)
        else:
            parameters = {}
        
        # Get available files
        csv_files = list(run_dir.glob('*.csv'))
        png_files = list(run_dir.glob('*.png'))
        
        return {
            'run_id': run_id,
            'summary': summary,
            'parameters': parameters,
            'csv_files': [f.name for f in csv_files],
            'png_files': [f.name for f in png_files],
            'run_dir': str(run_dir),
        }
    
    def get_run_image_base64(self, run_id, filename):
        """Get base64-encoded image for a specific run."""
        import base64
        run_dir = self.runs_dir / run_id
        img_path = run_dir / filename
        
        if not img_path.exists():
            return None
        
        with open(img_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode()
    
    def get_run_csv_data(self, run_id, filename):
        """Get CSV data for a specific run."""
        run_dir = self.runs_dir / run_id
        csv_path = run_dir / filename
        
        if not csv_path.exists():
            return None
        
        try:
            df = pd.read_csv(csv_path)
            return df.to_dict()
        except Exception as e:
            return None
    
    def delete_run(self, run_id):
        """Delete a simulation run and its results."""
        run_dir = self.runs_dir / run_id
        
        if run_dir.exists():
            shutil.rmtree(run_dir)
        
        # Remove from metadata
        self.metadata['runs'] = [r for r in self.metadata['runs'] if r['run_id'] != run_id]
        self._save_metadata()
        
        return True
    
    def compare_runs(self, run_ids):
        """Compare multiple runs side by side."""
        comparison = []
        
        for run_id in run_ids:
            run_info = self.get_run_details(run_id)
            if run_info:
                results = run_info['summary'].get('results', {})
                comparison.append({
                    'run_id': run_id,
                    'run_name': run_info['summary'].get('run_name', 'Unknown'),
                    'timestamp': run_info['summary'].get('timestamp', ''),
                    'optimal_ph': results.get('optimal_ph', 0),
                    'optimal_temp': results.get('optimal_temp', 0),
                    'h2_yield': results.get('optimal_h2_yield', 0),
                    'mpc_improvement': results.get('mpc_improvement', 0),
                    'lcoh': results.get('lcoh', 0),
                })

        return comparison
