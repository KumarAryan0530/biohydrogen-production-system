"""
Flask web application for Biohydrogen Production Dashboard.
Displays simulation results in an interactive, professional UI.
Includes parameter configuration and result versioning.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import sys
import subprocess
import shutil
import json
import threading
from pathlib import Path
from datetime import datetime

# Import AI module
try:
    import web_app.ai_insights as ai_insights
except ImportError:
    import ai_insights

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_app.data_loader import DataLoader
from simulation_manager import SimulationManager

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Initialize managers (relative to project root)
project_root = Path(__file__).parent.parent
results_dir = project_root / 'results'
data_loader = DataLoader(str(results_dir))
sim_manager = SimulationManager(str(project_root))


@app.route('/')
def landing():
    """Landing page for the application."""
    return render_template('landing.html')


@app.route('/dashboard')
def index():
    """Main dashboard page."""
    data = data_loader.get_dashboard_data()
    return render_template('index.html', data=data)


@app.route('/api/summary')
def api_summary():
    """API endpoint for summary data."""
    summary = data_loader.load_summary()
    return jsonify(summary)


@app.route('/api/sweep')
def api_sweep():
    """API endpoint for sweep results."""
    sweep = data_loader.load_sweep_results()
    if sweep is not None:
        return jsonify(sweep.to_dict())
    return jsonify({})


@app.route('/api/baseline')
def api_baseline():
    """API endpoint for baseline simulation."""
    baseline = data_loader.load_baseline_simulation()
    if baseline is not None:
        return jsonify({
            'time': baseline['time'].tolist() if 'time' in baseline.columns else [],
            'd_h2': baseline['d_h2'].tolist() if 'd_h2' in baseline.columns else [],
            'S_h2_cum': baseline['S_h2_cum'].tolist() if 'S_h2_cum' in baseline.columns else [],
        })
    return jsonify({})


@app.route('/api/mpc')
def api_mpc():
    """API endpoint for MPC control data."""
    mpc = data_loader.load_mpc_feed_rates()
    if mpc is not None:
        return jsonify(mpc.to_dict())
    return jsonify({})


@app.route('/detailed')
def detailed():
    """Detailed results page."""
    data = data_loader.get_dashboard_data()
    return render_template('detailed.html', data=data)


@app.route('/cost')
def cost():
    """Cost analysis page."""
    data = data_loader.get_dashboard_data()
    return render_template('cost.html', data=data)


@app.route('/sweep')
def sweep():
    """Parameter sweep page."""
    data = data_loader.get_dashboard_data()
    return render_template('sweep.html', data=data)


# ============================================================================
# NEW: Parameter Configuration & Simulation Management Routes
# ============================================================================

@app.route('/configure')
def configure():
    """Parameter configuration page for easy UI-based setup."""
    default_params = sim_manager.get_default_parameters()
    return render_template('configure.html', params=default_params)


@app.route('/api/live-logs')
def api_live_logs():
    """API endpoint to get live logs for the terminal."""
    log_file = project_root / 'results' / 'live_log.txt'
    if not log_file.exists():
        return jsonify({'logs': ''})
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return jsonify({'logs': f.read()})
    except Exception:
        return jsonify({'logs': ''})



@app.route('/api/run-simulation', methods=['POST'])
def api_run_simulation():
    """API endpoint to run simulation with custom parameters."""
    try:
        # Import here to avoid circular imports and numpy compatibility issues
        from main import run_with_parameters
        
        params = request.json
        run_name = params.get('run_name', 'Default_Run')
        
        # Create run directory
        run_id, run_dir = sim_manager.create_run(params)
        
        # Execute the simulation with custom parameters
        print(f"\n[API] Starting simulation for run: {run_id}")
        
        # Redirect stdout to capture logs
        old_stdout = sys.stdout
        log_file_path = project_root / 'results' / 'live_log.txt'
        sys.stdout = DualLogger(str(log_file_path))
        
        try:
            results = run_with_parameters(params)
        finally:
            if hasattr(sys.stdout, 'log'):
                sys.stdout.log.close()
            sys.stdout = old_stdout

        
        if results['status'] == 'error':
            return jsonify({
                'status': 'error',
                'message': results.get('error', 'Unknown error during simulation')
            }), 400
        
        # Copy results from results/ to the versioned run_dir
        results_dir = project_root / 'results'
        
        # Copy CSV files
        csv_files = ['baseline_simulation.csv', 'sweep_results.csv', 'mpc_feed_rates.csv', 'summary.csv']
        for csv_file in csv_files:
            src = results_dir / csv_file
            if src.exists():
                shutil.copy2(src, run_dir / csv_file)
        
        # Copy PNG files
        png_files = ['h2_yield_heatmap.png', 'baseline_h2_production.png', 'cost_breakdown.png']
        for png_file in png_files:
            src = results_dir / png_file
            if src.exists():
                shutil.copy2(src, run_dir / png_file)
        
        # Save parameters and summary to run directory
        with open(run_dir / 'parameters.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        # Create summary.json with metrics
        summary_data = {
            'run_id': run_id,
            'run_name': results['run_name'],
            'timestamp': datetime.now().isoformat(),
            'optimal_ph': results['optimal_ph'],
            'optimal_temp': results['optimal_temp'],
            'optimal_h2_yield': results['optimal_h2_yield'],
            'mpc_improvement': results['mpc_improvement'],
            'lcoh': results['lcoh'],
            'annual_h2_kg': results['annual_h2_kg'],
            'total_capital': results['total_capital']
        }
        
        with open(run_dir / 'summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Update metadata in runs_metadata.json
        sim_manager.save_run_results(run_id, run_dir, summary_data)
        
        return jsonify({
            'status': 'success',
            'message': f'Simulation completed successfully!',
            'run_id': run_id,
            'metrics': {
                'optimal_ph': round(results['optimal_ph'], 2),
                'optimal_temp': round(results['optimal_temp'], 1),
                'h2_yield': round(results['optimal_h2_yield'], 0),
                'mpc_improvement': round(results['mpc_improvement'], 1),
                'lcoh': round(results['lcoh'], 2)
            }
        }), 200
    
    except Exception as e:
        import traceback
        print(f"[API Error] {str(e)}")
        traceback.print_exc()
        
        # Get AI to diagnose the error if enabled
        ai_diagnosis = ai_insights.diagnose_error(str(e), params) if ai_insights.is_ai_enabled() else ""
        
        return jsonify({
            'status': 'error',
            'message': str(e),
            'ai_diagnosis': ai_diagnosis
        }), 400


@app.route('/history')
def history():
    """View all past simulation runs and their results."""
    runs = sim_manager.get_all_runs()
    
    # Format for display
    runs_display = []
    for run in runs:
        runs_display.append({
            'run_id': run['run_id'],
            'run_name': run['run_name'],
            'timestamp': run['timestamp'],
            'optimal_ph': f"{run.get('optimal_ph', 0):.2f}",
            'optimal_temp': f"{run.get('optimal_temp', 0):.1f}",
            'h2_yield': f"{run.get('optimal_h2_yield', 0):,.0f}",
            'mpc_improvement': f"{run.get('mpc_improvement', 0):.1f}",
            'lcoh': f"${run.get('lcoh', 0):.2f}",
            'raw_ph': run.get('optimal_ph', 0),
            'raw_mpc': run.get('mpc_improvement', 0),
            'raw_lcoh': run.get('lcoh', 0),
            'raw_yield': run.get('optimal_h2_yield', 0)
        })
    
    return render_template('history.html', runs=runs_display)


@app.route('/results/<run_id>')
def result_details(run_id):
    """View detailed results for a specific run."""
    run_info = sim_manager.get_run_details(run_id)
    
    if not run_info:
        return render_template('404.html', message='Run not found'), 404
    
    # Get images as base64
    images = {}
    for png_file in ['h2_yield_heatmap.png', 'baseline_h2_production.png', 'cost_breakdown.png']:
        img_data = sim_manager.get_run_image_base64(run_id, png_file)
        if img_data:
            images[png_file] = img_data
    
    # Get CSV data
    csv_data = {}
    for csv_file in run_info['csv_files']:
        data = sim_manager.get_run_csv_data(run_id, csv_file)
        if data:
            csv_data[csv_file] = data
    
    # Generate AI insight if it doesn't exist and AI is enabled
    summary_data = run_info.get('summary', {})
    existing_insight = summary_data.get('ai_insight', '')
    
    if (not existing_insight or 'Error generating' in existing_insight) and ai_insights.is_ai_enabled():
        results_data = summary_data.get('results', {})
        summary_data['ai_insight'] = ai_insights.generate_executive_summary({**results_data})
        # Save it back to summary.json
        summary_file = Path(run_info['run_dir']) / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

    return render_template('result_details.html',
                         run_info=run_info,
                         images=images,
                         csv_data=csv_data)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for Biohydrogen Assistant chat."""
    data = request.json
    run_id = data.get('run_id')
    chat_history = data.get('chat_history', [])
    new_message = data.get('message', '')
    
    run_info = sim_manager.get_run_details(run_id)
    if not run_info:
        return jsonify({'error': 'Run not found'}), 404
        
    results_data = run_info.get('summary', {}).get('results', {})
    response_text = ai_insights.chat_with_run({**results_data}, chat_history, new_message)
    
    return jsonify({'response': response_text}), 200


@app.route('/api/delete-run/<run_id>', methods=['DELETE'])
def api_delete_run(run_id):
    """API endpoint to delete a run."""
    try:
        sim_manager.delete_run(run_id)
        return jsonify({
            'status': 'success',
            'message': f'Run {run_id} deleted'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/compare-runs', methods=['POST'])
def api_compare_runs():
    """API endpoint to compare multiple runs."""
    try:
        run_ids = request.json.get('run_ids', [])
        comparison = sim_manager.compare_runs(run_ids)
        
        # Generate AI Summary for the Comparison
        try:
            ai_summary = ai_insights.compare_runs_ai(comparison) if ai_insights.is_ai_enabled() else "AI insights disabled or not configured."
        except Exception as e:
            ai_summary = "AI Summary unavailable: " + str(e)
            
        return jsonify({
            'status': 'success',
            'comparison': comparison,
            'ai_summary': ai_summary
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
