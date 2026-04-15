# 🚀 Quick Start Commands - Biohydrogen Project

## **1️⃣ SETUP (First Time Only)**

### Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## **2️⃣ RUN WEB DASHBOARD**

### Start Web Server
```powershell
.\venv\Scripts\Activate.ps1
python run_web.py
```

### Access Dashboard
Open browser: **http://localhost:5000**

---

## **3️⃣ WEB INTERFACE USAGE**

### Pages Available
```
http://localhost:5000/                  → Main Dashboard
http://localhost:5000/configure         → Create New Simulation
http://localhost:5000/history           → View All Runs
http://localhost:5000/results/<run_id>  → View Specific Run
```

---

## **4️⃣ RUN SIMULATION FROM TERMINAL**

### Run with Default Parameters
```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

### Run with Custom Parameters (Edit main.py first)
```powershell
.\venv\Scripts\Activate.ps1
python -c "from main import run_with_parameters; run_with_parameters({'run_name': 'My_Run', 'baseline_ph': 5.5, 'baseline_temp': 35})"
```

---

## **5️⃣ FILE LOCATIONS**

### Latest Results (Overwritten Each Run)
```
results/
├── summary.csv
├── baseline_simulation.csv
├── sweep_results.csv
├── mpc_feed_rates.csv
├── h2_yield_heatmap.png
├── baseline_h2_production.png
└── cost_breakdown.png
```

### Saved Results (Never Overwritten)
```
results_runs/
├── runs_metadata.json
├── 20260412_164727_Default_Run/
├── 20260412_170000_My_Experiment/
└── ...
```

---

## **6️⃣ CONFIGURE SIMULATION**

### Using Web Form (RECOMMENDED)
1. Go to: **http://localhost:5000/configure**
2. Fill in parameters
3. Click "Submit"
4. Wait 1-2 minutes
5. Results auto-save and display

### Using Configure Page Parameters
```
• Run Name: Name your experiment
• Baseline pH: 5.0-8.0
• Baseline Temperature: 25-40°C
• Sweep ranges: pH min/max, Temp min/max
• MPC Control: Horizon, Interval, Time
• Economics: Capital cost, Plant life, Discount rate
```

---

## **7️⃣ VIEW RESULTS**

### View All Simulations
```
http://localhost:5000/history
```

### View Specific Run
Click on any run in History → View full details

### Download Raw Data
Go to run details → "Download Raw Data" section

---

## **8️⃣ PROJECT STRUCTURE**

```
Biohydrogen/
├── main.py                    ← Main simulation script
├── run_web.py                 ← Start web server
├── requirements.txt           ← Python dependencies
├── simulation_manager.py       ← Manage runs & versioning
├── web_app/
│   ├── app.py                 ← Flask routes
│   ├── data_loader.py         ← Load CSV/images
│   ├── templates/             ← HTML pages
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── configure.html
│   │   ├── history.html
│   │   └── result_details.html
│   └── static/css/style.css
├── simulation/
│   └── adm1_biohydrogen.py    ← Simulation model
├── optimization/
│   └── parameter_sweep.py     ← Optimization algorithms
├── control/
│   └── fermentation_control.py ← MPC control
├── economics/
│   └── h2_cost.py             ← Cost analysis
├── results/                   ← Latest results
└── results_runs/              ← Saved results history
```

---

## **9️⃣ TROUBLESHOOTING**

### Web Server Won't Start
```powershell
# Kill any running server
Get-Process python | Stop-Process -Force

# Restart
.\venv\Scripts\Activate.ps1
python run_web.py
```

### Dependencies Issue
```powershell
.\venv\Scripts\Activate.ps1
pip install --upgrade -r requirements.txt
```

### Clear Old Results
```powershell
rm -r results_runs\*_Old_Run
```

---

## **🔟 USEFUL SHORTCUTS**

### Stop Web Server
```
Press: Ctrl + C (in terminal)
```

### Clear Results
```powershell
rm results\*.csv
rm results\*.png
```

### View Metadata
```powershell
cat results_runs\runs_metadata.json
```

### Check Installed Packages
```powershell
.\venv\Scripts\Activate.ps1
pip list
```

---

## **WORKFLOW EXAMPLE**

```
1. Start server
   → .\venv\Scripts\Activate.ps1
   → python run_web.py

2. Open browser
   → http://localhost:5000

3. Configure simulation
   → Click "Configure"
   → Fill form
   → Submit

4. Wait for completion
   → Status: Running... (auto-refreshes)
   → Status: Complete!

5. View results
   → Automatically goes to History
   → Click run name
   → See all metrics & charts

6. Download data
   → In result details
   → Click "Download Raw Data"
```

---

## **QUICK REFERENCE**

| Task | Command |
|------|---------|
| Activate venv | `.\venv\Scripts\Activate.ps1` |
| Install packages | `pip install -r requirements.txt` |
| Start server | `python run_web.py` |
| Run simulation | `python main.py` |
| View dashboard | `http://localhost:5000` |
| Configure run | `http://localhost:5000/configure` |
| View history | `http://localhost:5000/history` |
| Stop server | `Ctrl + C` |

---

**Created:** April 12, 2026  
**For:** Biohydrogen Production System  
**Last Updated:** Today
