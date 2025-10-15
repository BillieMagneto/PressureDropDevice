#!/usr/bin/env python3
"""
Standalone Water and Air Data Analysis GUI
Complete application in a single file for easier use
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import optimize
from sklearn.metrics import r2_score
import warnings
import os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


class WaterAirAnalyzer:
    """Standalone GUI Application for Water and Air Data Analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Water and Air Data Analysis - Point Cloud Analyzer")
        self.root.geometry("1600x1000")
        
        # Data storage
        self.water_data = None
        self.air_data = None
        self.water_fit_results = None
        self.air_fit_results = None
        self.relationship_results = None
        
        # Default dataset configuration
        self.default_dataset_config = None
        self.default_dataset_file = "default_dataset.json"
        
        # File name storage for automatic experiment naming
        self.water_file_name = None
        self.air_file_name = None
        
        # Experiment storage for comparison
        self.experiments = []  # List to store multiple experiments
        self.current_experiment_name = None
        self.relationship_results = None
        
        # Column selections
        self.water_x_col = tk.StringVar()
        self.water_y_col = tk.StringVar()
        self.air_x_col = tk.StringVar()
        self.air_y_col = tk.StringVar()
        
        self.create_interface()
        
    def create_interface(self):
        """Create the main interface"""
        self.create_menu()
        self.create_main_panels()
        self.create_status_bar()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Archivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Water Data...", command=self.import_water_data)
        file_menu.add_command(label="Import Air Data...", command=self.import_air_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save as Default Dataset", command=self.save_default_dataset)
        file_menu.add_command(label="Load Default Dataset", command=self.load_default_dataset)
        file_menu.add_command(label="Clear Default Dataset", command=self.clear_default_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Export Analysis...", command=self.export_analysis)
        file_menu.add_command(label="Export Current Summary...", command=self.export_current_summary)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Análisis
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Fit Curves", command=self.fit_curves)
        analysis_menu.add_command(label="Find Relationship", command=self.find_relationship)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="View Linear Relationships Only", command=self.view_linear_relationships)
        analysis_menu.add_command(label="Calculate Average Linear Relationship", command=self.calculate_average_linear_relationship)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="View Polynomial Degree 2 Relationships", command=self.view_polynomial2_relationships)
        analysis_menu.add_command(label="Calculate Average Polynomial Degree 2 Relationship", command=self.calculate_average_polynomial2_relationship)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="View Polynomial Degree 3 Relationships", command=self.view_polynomial3_relationships)
        analysis_menu.add_command(label="Calculate Average Polynomial Degree 3 Relationship", command=self.calculate_average_polynomial3_relationship)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Clear Results", command=self.clear_results)
        
        # Experiments menu
        experiments_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Experiments", menu=experiments_menu)
        experiments_menu.add_command(label="Save Current Experiment", command=self.save_experiment)
        experiments_menu.add_command(label="Load Experiment", command=self.load_experiment)
        experiments_menu.add_separator()
        experiments_menu.add_command(label="Compare Experiments", command=self.compare_experiments)
        experiments_menu.add_command(label="View Experiment List", command=self.view_experiments)
        experiments_menu.add_separator()
        experiments_menu.add_command(label="Export All Experiments", command=self.export_all_experiments)
        experiments_menu.add_command(label="Export Summary Report", command=self.export_summary_report)
        experiments_menu.add_separator()
        experiments_menu.add_command(label="Test Calculated Columns", command=self.test_calculated_columns)
        experiments_menu.add_command(label="Debug Air Linear Calculation", command=self.debug_air_linear_calculation)
        
        # Ayuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_panels(self):
        """Create main working panels"""
        # Main container with three sections
        main_notebook = ttk.Notebook(self.root)
        main_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Data Import and Configuration
        import_frame = ttk.Frame(main_notebook)
        main_notebook.add(import_frame, text="1. Import Data")
        
        # Tab 2: Analysis and Fitting
        analysis_frame = ttk.Frame(main_notebook)
        main_notebook.add(analysis_frame, text="2. Analysis & Fitting")
        
        # Tab 3: Results and Relationships
        results_frame = ttk.Frame(main_notebook)
        main_notebook.add(results_frame, text="3. Results & Relationships")
        
        self.create_import_panel(import_frame)
        self.create_analysis_panel(analysis_frame)
        self.create_results_panel(results_frame)
        
    def create_import_panel(self, parent):
        """Create data import panel"""
        # Split into left and right for water and air
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Water data section
        water_frame = ttk.LabelFrame(paned_window, text="Water Data")
        paned_window.add(water_frame, weight=1)
        
        # Air data section
        air_frame = ttk.LabelFrame(paned_window, text="Air Data")
        paned_window.add(air_frame, weight=1)
        
        self.create_data_section(water_frame, "water")
        self.create_data_section(air_frame, "air")
        
    def create_data_section(self, parent, data_type):
        """Create a data section for water or air"""
        # Import button
        import_btn = ttk.Button(parent, 
                               text=f"📁 Import Excel - {data_type.title()}", 
                               command=lambda: self.import_data(data_type))
        import_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # File info
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="File:").pack(anchor=tk.W)
        file_label = ttk.Label(info_frame, text="No file selected", 
                              foreground="gray")
        file_label.pack(anchor=tk.W)
        
        # Store reference to label
        if data_type == "water":
            self.water_file_label = file_label
        else:
            self.air_file_label = file_label
            
        # Column selection
        col_frame = ttk.LabelFrame(parent, text="Column Selection")
        col_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # X column
        ttk.Label(col_frame, text="X Column:").pack(anchor=tk.W, padx=5)
        x_combo = ttk.Combobox(col_frame, 
                              textvariable=self.water_x_col if data_type == "water" else self.air_x_col)
        x_combo.pack(fill=tk.X, padx=5, pady=2)
        x_combo.bind('<<ComboboxSelected>>', lambda e: self.update_data_preview(data_type))
        
        # Y column
        ttk.Label(col_frame, text="Y Column:").pack(anchor=tk.W, padx=5)
        y_combo = ttk.Combobox(col_frame, 
                              textvariable=self.water_y_col if data_type == "water" else self.air_y_col)
        y_combo.pack(fill=tk.X, padx=5, pady=2)
        y_combo.bind('<<ComboboxSelected>>', lambda e: self.update_data_preview(data_type))
        
        # Store references
        if data_type == "water":
            self.water_x_combo = x_combo
            self.water_y_combo = y_combo
        else:
            self.air_x_combo = x_combo
            self.air_y_combo = y_combo
            
        # Data preview
        preview_frame = ttk.LabelFrame(parent, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for data preview
        columns = ('Index', 'X', 'Y')
        tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80)
            
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store reference
        if data_type == "water":
            self.water_tree = tree
        else:
            self.air_tree = tree
            
    def create_analysis_panel(self, parent):
        """Create analysis and fitting panel"""
        # Split into controls and visualization
        paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls panel
        controls_frame = ttk.Frame(paned_window)
        paned_window.add(controls_frame, weight=1)
        
        # Visualization panel
        viz_frame = ttk.Frame(paned_window)
        paned_window.add(viz_frame, weight=2)
        
        # Model selection
        model_frame = ttk.LabelFrame(controls_frame, text="Curve Fitting Models")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_vars = {}
        models = [
            ("Linear", "linear"),
            ("Polynomial (degree 2)", "poly2"),
            ("Polynomial (degree 3)", "poly3"),
            ("Exponential", "exponential"),
            ("Logarithmic", "logarithmic"),
            ("Power", "power")
        ]
        
        for display_name, model_name in models:
            var = tk.BooleanVar(value=True)  # Select all by default
            self.model_vars[model_name] = var
            ttk.Checkbutton(model_frame, text=display_name, variable=var).pack(anchor=tk.W, padx=5)
            
        # Analysis buttons
        button_frame = ttk.LabelFrame(controls_frame, text="Analysis")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="🔄 Fit Curves", 
                  command=self.fit_curves).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="🔗 Find Relationship", 
                  command=self.find_relationship).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(button_frame, text="📊 Update Plot", 
                  command=self.update_plots).pack(fill=tk.X, padx=5, pady=2)
        
        # Linear analysis buttons
        linear_frame = ttk.LabelFrame(controls_frame, text="Linear Analysis")
        linear_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(linear_frame, text="📈 View Linear Relationships", 
                  command=self.view_linear_relationships).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(linear_frame, text="🧮 Calculate Average Relationship", 
                  command=self.calculate_average_linear_relationship).pack(fill=tk.X, padx=5, pady=2)
        
        # Polynomial analysis buttons
        poly_frame = ttk.LabelFrame(controls_frame, text="Polynomial Analysis")
        poly_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(poly_frame, text="📊 View Poly Degree 2 Relationships", 
                  command=self.view_polynomial2_relationships).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(poly_frame, text="🔢 Average Poly Degree 2 Relationship", 
                  command=self.calculate_average_polynomial2_relationship).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(poly_frame, text="📉 View Poly Degree 3 Relationships", 
                  command=self.view_polynomial3_relationships).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(poly_frame, text="🎯 Average Poly Degree 3 Relationship", 
                  command=self.calculate_average_polynomial3_relationship).pack(fill=tk.X, padx=5, pady=2)
        
        # Results summary
        summary_frame = ttk.LabelFrame(controls_frame, text="Results Summary")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=15, wrap=tk.WORD)
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, 
                                        command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create visualization area
        self.create_visualization_area(viz_frame)
        
    def create_visualization_area(self, parent):
        """Create the visualization area with plots"""
        viz_notebook = ttk.Notebook(parent)
        viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Individual plots tab
        individual_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(individual_frame, text="Individual Fits")
        
        # Comparison tab
        comparison_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(comparison_frame, text="Comparison & Relationship")
        
        # Create matplotlib figures
        self.fig_individual = Figure(figsize=(12, 8), dpi=100)
        self.canvas_individual = FigureCanvasTkAgg(self.fig_individual, individual_frame)
        self.canvas_individual.draw()
        self.canvas_individual.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_comparison = Figure(figsize=(12, 8), dpi=100)
        self.canvas_comparison = FigureCanvasTkAgg(self.fig_comparison, comparison_frame)
        self.canvas_comparison.draw()
        self.canvas_comparison.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_results_panel(self, parent):
        """Create results and export panel"""
        # Results display
        results_frame = ttk.LabelFrame(parent, text="Detailed Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, font=('Courier', 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                        command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export controls
        export_frame = ttk.LabelFrame(parent, text="Export Results")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        export_buttons = ttk.Frame(export_frame)
        export_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_buttons, text="💾 Export to Excel", 
                  command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_buttons, text="📊 Save Plots", 
                  command=self.save_plots).pack(side=tk.LEFT, padx=5)
        
    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Import water and air data to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Auto-load default dataset if available
        self.root.after(100, self.auto_load_default_dataset)  # Load after GUI is ready
        
    # Data import methods
    def import_water_data(self):
        """Import water data specifically"""
        self.import_data("water")
        
    def import_air_data(self):
        """Import air data specifically"""
        self.import_data("air")
        
    def import_data(self, data_type):
        """Import data from Excel file"""
        file_path = filedialog.askopenfilename(
            title=f"Select Excel file - {data_type.title()} Data",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Loading {data_type} data...")
                
                # Load data based on file extension
                file_ext = Path(file_path).suffix.lower()
                if file_ext == '.csv':
                    data = pd.read_csv(file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    data = pd.read_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                # Clean data
                data = data.dropna(how='all')  # Remove empty rows
                data = data.dropna(axis=1, how='all')  # Remove empty columns
                
                # Convert to numeric where possible
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        pass
                
                if data_type == "water":
                    self.water_data = data
                    self.water_file_name = Path(file_path).stem
                    self.last_water_file_path = file_path  # Store file path for default dataset
                    self.water_file_label.config(text=self.water_file_name)
                    self.update_column_combos("water", data.columns.tolist())
                    self.update_data_preview("water")
                else:
                    self.air_data = data
                    self.air_file_name = Path(file_path).stem
                    self.last_air_file_path = file_path  # Store file path for default dataset
                    self.air_file_label.config(text=self.air_file_name)
                    self.update_column_combos("air", data.columns.tolist())
                    self.update_data_preview("air")
                    
                self.status_var.set(f"{data_type.title()} data loaded successfully")
                
            except Exception as e:
                messagebox.showerror("File Loading Error", 
                                   f"Could not load the file:\n{str(e)}")
                self.status_var.set("Error loading data")
                
    def update_column_combos(self, data_type, columns):
        """Update column selection combos"""
        if data_type == "water":
            self.water_x_combo['values'] = columns
            self.water_y_combo['values'] = columns
            if columns:
                self.water_x_combo.current(0)
                if len(columns) > 1:
                    self.water_y_combo.current(1)
        else:
            self.air_x_combo['values'] = columns
            self.air_y_combo['values'] = columns
            if columns:
                self.air_x_combo.current(0)
                if len(columns) > 1:
                    self.air_y_combo.current(1)
                    
    def update_data_preview(self, data_type):
        """Update data preview in treeview"""
        if data_type == "water":
            tree = self.water_tree
            data = self.water_data
            x_col = self.water_x_col.get()
            y_col = self.water_y_col.get()
        else:
            tree = self.air_tree
            data = self.air_data
            x_col = self.air_x_col.get()
            y_col = self.air_y_col.get()
            
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
            
        if data is not None and x_col and y_col and x_col in data.columns and y_col in data.columns:
            # Show first 20 rows
            for i, (x_val, y_val) in enumerate(zip(data[x_col].head(20), data[y_col].head(20))):
                tree.insert('', 'end', values=(i+1, f"{x_val:.4f}" if pd.notna(x_val) else "N/A", 
                                             f"{y_val:.4f}" if pd.notna(y_val) else "N/A"))
                                             
    # Analysis methods
    def fit_curves(self):
        """Fit curves to both water and air data"""
        if self.water_data is None or self.air_data is None:
            messagebox.showwarning("Incomplete Data", 
                                 "Please import both water and air data")
            return
            
        self.status_var.set("Fitting curves...")
        
        try:
            # Fit water data
            self.water_fit_results = self.fit_data_curves(self.water_data, 
                                                        self.water_x_col.get(), 
                                                        self.water_y_col.get(), 
                                                        "Water")
            
            # Fit air data
            self.air_fit_results = self.fit_data_curves(self.air_data, 
                                                       self.air_x_col.get(), 
                                                       self.air_y_col.get(), 
                                                       "Air")
            
            self.update_summary()
            self.update_plots()
            self.status_var.set("Curve fitting completed")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during fitting:\n{str(e)}")
            self.status_var.set("Error in analysis")
            
    def fit_data_curves(self, data, x_col, y_col, data_name):
        """Fit multiple curve models to data"""
        if x_col not in data.columns or y_col not in data.columns:
            raise ValueError(f"Columns not found in {data_name} data")
            
        # Extract and clean data
        valid_data = data[[x_col, y_col]].dropna()
        x_data = valid_data[x_col].values
        y_data = valid_data[y_col].values
        
        if len(x_data) < 3:
            raise ValueError(f"Insufficient data for {data_name} (minimum 3 points)")
        
        results = {}
        
        # Fit selected models
        for model_name, var in self.model_vars.items():
            if not var.get():
                continue
                
            try:
                if model_name == "linear":
                    popt, r2, func = self.fit_linear(x_data, y_data)
                elif model_name == "poly2":
                    popt, r2, func = self.fit_polynomial(x_data, y_data, 2)
                elif model_name == "poly3":
                    popt, r2, func = self.fit_polynomial(x_data, y_data, 3)
                elif model_name == "exponential":
                    popt, r2, func = self.fit_exponential(x_data, y_data)
                elif model_name == "logarithmic":
                    popt, r2, func = self.fit_logarithmic(x_data, y_data)
                elif model_name == "power":
                    popt, r2, func = self.fit_power(x_data, y_data)
                else:
                    continue
                    
                results[model_name] = {
                    'params': popt,
                    'r2': r2,
                    'function': func,
                    'x_data': x_data,
                    'y_data': y_data
                }
                
            except Exception as e:
                print(f"Error fitting {model_name} to {data_name}: {e}")
                continue
                
        return results
        
    # Curve fitting functions
    def fit_linear(self, x, y):
        """Fit linear model: y = ax + b"""
        popt = np.polyfit(x, y, 1)
        y_pred = np.polyval(popt, x)
        r2 = r2_score(y, y_pred)
        
        def func(x_val):
            return popt[0] * x_val + popt[1]
            
        return popt, r2, func
        
    def fit_polynomial(self, x, y, degree):
        """Fit polynomial model"""
        popt = np.polyfit(x, y, degree)
        y_pred = np.polyval(popt, x)
        r2 = r2_score(y, y_pred)
        
        def func(x_val):
            return np.polyval(popt, x_val)
            
        return popt, r2, func
        
    def fit_exponential(self, x, y):
        """Fit exponential model: y = a * exp(b * x)"""
        try:
            # Ensure positive y values for log transformation
            y_pos = np.abs(y) + 1e-10
            log_y = np.log(y_pos)
            
            popt = np.polyfit(x, log_y, 1)
            a = np.exp(popt[1])
            b = popt[0]
            
            def func(x_val):
                return a * np.exp(b * x_val)
                
            y_pred = func(x)
            r2 = r2_score(y, y_pred)
            
            return [a, b], r2, func
            
        except:
            # Fallback to direct optimization
            def exp_func(x_val, a, b):
                return a * np.exp(b * x_val)
                
            popt, _ = optimize.curve_fit(exp_func, x, y, p0=[1, 0.1], maxfev=1000)
            
            def func(x_val):
                return exp_func(x_val, *popt)
                
            y_pred = func(x)
            r2 = r2_score(y, y_pred)
            
            return popt, r2, func
            
    def fit_logarithmic(self, x, y):
        """Fit logarithmic model: y = a * ln(x) + b"""
        x_pos = np.abs(x) + 1e-10  # Ensure positive
        log_x = np.log(x_pos)
        
        popt = np.polyfit(log_x, y, 1)
        
        def func(x_val):
            return popt[0] * np.log(np.abs(x_val) + 1e-10) + popt[1]
            
        y_pred = func(x)
        r2 = r2_score(y, y_pred)
        
        return popt, r2, func
        
    def fit_power(self, x, y):
        """Fit power model: y = a * x^b"""
        try:
            # Transform to linear: ln(y) = ln(a) + b*ln(x)
            x_pos = np.abs(x) + 1e-10
            y_pos = np.abs(y) + 1e-10
            
            log_x = np.log(x_pos)
            log_y = np.log(y_pos)
            
            popt = np.polyfit(log_x, log_y, 1)
            a = np.exp(popt[1])
            b = popt[0]
            
            def func(x_val):
                return a * np.power(np.abs(x_val) + 1e-10, b)
                
            y_pred = func(x)
            r2 = r2_score(y, y_pred)
            
            return [a, b], r2, func
            
        except:
            # Fallback
            def power_func(x_val, a, b):
                return a * np.power(np.abs(x_val) + 1e-10, b)
                
            popt, _ = optimize.curve_fit(power_func, x, y, p0=[1, 1], maxfev=1000)
            
            def func(x_val):
                return power_func(x_val, *popt)
                
            y_pred = func(x)
            r2 = r2_score(y, y_pred)
            
            return popt, r2, func
            
    def find_relationship(self):
        """Find relationship between water and air data"""
        if self.water_fit_results is None or self.air_fit_results is None:
            messagebox.showwarning("Incomplete Analysis", 
                                 "Please run curve fitting first")
            return
            
        self.status_var.set("Finding relationship between water and air...")
        
        try:
            # Get best fits for both datasets
            water_best = self.get_best_fit(self.water_fit_results)
            air_best = self.get_best_fit(self.air_fit_results)
            
            if water_best is None or air_best is None:
                messagebox.showwarning("No Results", 
                                     "No valid fits found")
                return
                
            # Find relationship between the two fitted functions
            self.relationship_results = self.analyze_relationship(water_best, air_best)
            
            self.update_summary()
            self.update_plots()
            self.status_var.set("Relationship analysis completed")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error in relationship analysis:\n{str(e)}")
            self.status_var.set("Error in relationship analysis")
            
    def get_best_fit(self, fit_results):
        """Get the best fitting model based on R²"""
        if not fit_results:
            return None
            
        best_model = None
        best_r2 = -1
        
        for model_name, results in fit_results.items():
            if results['r2'] > best_r2:
                best_r2 = results['r2']
                best_model = results
                best_model['name'] = model_name
                
        return best_model
        
    def analyze_relationship(self, water_fit, air_fit):
        """Analyze relationship between water and air fitted curves"""
        # Generate comparison points
        x_min = max(np.min(water_fit['x_data']), np.min(air_fit['x_data']))
        x_max = min(np.max(water_fit['x_data']), np.max(air_fit['x_data']))
        
        x_comparison = np.linspace(x_min, x_max, 100)
        
        water_y = water_fit['function'](x_comparison)
        air_y = air_fit['function'](x_comparison)
        
        # Calculate ratio and difference
        ratio = air_y / (water_y + 1e-10)  # Avoid division by zero
        difference = air_y - water_y
        
        # Fit relationship models
        relationships = {}
        
        # 1. Linear relationship: y_water = a * y_air + b (inverse relationship)
        try:
            # Fit Y_air = a * Y_water + b, then solve for Y_water = (Y_air - b) / a
            popt_linear = np.polyfit(water_y, air_y, 1)
            y_pred = np.polyval(popt_linear, water_y)
            r2_linear = r2_score(air_y, y_pred)
            
            # Convert to inverse relationship: Y_water = (Y_air - b) / a
            a, b = popt_linear
            if abs(a) > 1e-10:  # Avoid division by zero
                # Y_water = (1/a) * Y_air - (b/a)
                a_inv = 1.0 / a
                b_inv = -b / a
                relationships['linear'] = {
                    'params': [a_inv, b_inv],
                    'r2': r2_linear,
                    'equation': f"Y_water = {a_inv:.4f} * Y_air + {b_inv:.4f}",
                    'type': 'linear'
                }
        except:
            pass
        
        # 2. Polynomial relationship: Y_water = a * Y_air² + b * Y_air + c
        try:
            # Fit Y_air = a * Y_water² + b * Y_water + c, then find best inverse
            popt_poly = np.polyfit(water_y, air_y, 2)
            y_pred_poly = np.polyval(popt_poly, water_y)
            r2_poly = r2_score(air_y, y_pred_poly)
            
            # For inverse, we'll use direct fitting Y_water vs Y_air (polynomial)
            popt_inv_poly = np.polyfit(air_y, water_y, 2)
            a_inv, b_inv, c_inv = popt_inv_poly
            
            relationships['polynomial'] = {
                'params': popt_inv_poly,
                'r2': r2_poly,
                'equation': f"Y_water = {a_inv:.4f} * Y_air² + {b_inv:.4f} * Y_air + {c_inv:.4f}",
                'type': 'polynomial'
            }
        except:
            pass
        
        # 3. Power relationship: Y_water = a * Y_air^b
        try:
            # Filter positive values for log transformation
            mask = (air_y > 0) & (water_y > 0)
            if np.sum(mask) > 5:  # Need enough points
                air_y_pos = air_y[mask]
                water_y_pos = water_y[mask]
                
                # Transform to linear: log(Y_water) = log(a) + b * log(Y_air)
                log_air = np.log(air_y_pos)
                log_water = np.log(water_y_pos)
                
                popt_power = np.polyfit(log_air, log_water, 1)
                b_power, log_a = popt_power
                a_power = np.exp(log_a)
                
                # Calculate R²
                y_pred_power = a_power * (air_y_pos ** b_power)
                r2_power = r2_score(water_y_pos, y_pred_power)
                
                relationships['power'] = {
                    'params': [a_power, b_power],
                    'r2': r2_power,
                    'equation': f"Y_water = {a_power:.4f} * Y_air^{b_power:.4f}",
                    'type': 'power'
                }
        except:
            pass
        
        # 4. Exponential relationship: Y_water = a * exp(b * Y_air)
        try:
            # Filter positive water values for log transformation
            mask = water_y > 0
            if np.sum(mask) > 5:
                air_y_pos = air_y[mask]
                water_y_pos = water_y[mask]
                
                # Transform to linear: log(Y_water) = log(a) + b * Y_air
                log_water = np.log(water_y_pos)
                
                popt_exp = np.polyfit(air_y_pos, log_water, 1)
                b_exp, log_a = popt_exp
                a_exp = np.exp(log_a)
                
                # Calculate R²
                y_pred_exp = a_exp * np.exp(b_exp * air_y_pos)
                r2_exp = r2_score(water_y_pos, y_pred_exp)
                
                relationships['exponential'] = {
                    'params': [a_exp, b_exp],
                    'r2': r2_exp,
                    'equation': f"Y_water = {a_exp:.4f} * exp({b_exp:.4f} * Y_air)",
                    'type': 'exponential'
                }
        except:
            pass
        
        # 5. Logarithmic relationship: Y_water = a * log(Y_air) + b
        try:
            # Filter positive air values for log transformation
            mask = air_y > 0
            if np.sum(mask) > 5:
                air_y_pos = air_y[mask]
                water_y_pos = water_y[mask]
                
                # Y_water = a * log(Y_air) + b
                log_air = np.log(air_y_pos)
                
                popt_log = np.polyfit(log_air, water_y_pos, 1)
                a_log, b_log = popt_log
                
                # Calculate R²
                y_pred_log = a_log * log_air + b_log
                r2_log = r2_score(water_y_pos, y_pred_log)
                
                relationships['logarithmic'] = {
                    'params': [a_log, b_log],
                    'r2': r2_log,
                    'equation': f"Y_water = {a_log:.4f} * log(Y_air) + {b_log:.4f}",
                    'type': 'logarithmic'
                }
        except:
            pass
            
        # Find best relationship based on R²
        best_relationship = None
        best_r2 = -1
        
        for rel_name, rel_data in relationships.items():
            if rel_name != 'ratio' and 'r2' in rel_data:
                if rel_data['r2'] > best_r2:
                    best_r2 = rel_data['r2']
                    best_relationship = {
                        'name': rel_name,
                        'data': rel_data
                    }
            
        # Ratio analysis (inverse relationship)
        valid_ratio = ratio[np.isfinite(ratio)]
        if len(valid_ratio) > 0:
            mean_ratio = np.mean(valid_ratio)
            std_ratio = np.std(valid_ratio)
            
            # Convert to inverse ratio: Y_water / Y_air
            if abs(mean_ratio) > 1e-10:  # Avoid division by zero
                mean_ratio_inv = 1.0 / mean_ratio
                std_ratio_inv = std_ratio / (mean_ratio ** 2)  # Error propagation
                
                relationships['ratio'] = {
                    'mean': mean_ratio_inv,
                    'std': std_ratio_inv,
                    'equation': f"Y_water ≈ {mean_ratio_inv:.4f} * Y_air (±{std_ratio_inv:.4f})",
                    'type': 'ratio'
                }
            else:
                relationships['ratio'] = {
                    'mean': mean_ratio,
                    'std': std_ratio,
                    'equation': f"Y_air ≈ {mean_ratio:.4f} * Y_water (±{std_ratio:.4f}) (cannot invert)"
                }
        
        return {
            'water_model': water_fit['name'],
            'air_model': air_fit['name'],
            'x_range': (x_min, x_max),
            'water_y': water_y,
            'air_y': air_y,
            'relationships': relationships,
            'comparison_x': x_comparison
        }
        
    def update_summary(self):
        """Update summary text with results"""
        self.summary_text.delete(1.0, tk.END)
        
        summary = "=== ANALYSIS SUMMARY ===\n\n"
        
        if self.water_fit_results:
            summary += "WATER DATA:\n"
            for model_name, results in self.water_fit_results.items():
                summary += f"  {model_name}: R² = {results['r2']:.4f}\n"
            best_water = self.get_best_fit(self.water_fit_results)
            if best_water:
                summary += f"  Best fit: {best_water['name']} (R² = {best_water['r2']:.4f})\n\n"
            
        if self.air_fit_results:
            summary += "AIR DATA:\n"
            for model_name, results in self.air_fit_results.items():
                summary += f"  {model_name}: R² = {results['r2']:.4f}\n"
            best_air = self.get_best_fit(self.air_fit_results)
            if best_air:
                summary += f"  Best fit: {best_air['name']} (R² = {best_air['r2']:.4f})\n\n"
            
        if self.relationship_results:
            summary += "WATER-AIR RELATIONSHIPS:\n"
            summary += f"  Water Model: {self.relationship_results['water_model']}\n"
            summary += f"  Air Model: {self.relationship_results['air_model']}\n\n"
            
            # Sort relationships by R² (highest first)
            relationships = self.relationship_results['relationships']
            sorted_rels = []
            
            for rel_name, rel_data in relationships.items():
                if rel_name != 'ratio' and 'r2' in rel_data:
                    sorted_rels.append((rel_name, rel_data))
            
            # Sort by R² descending
            sorted_rels.sort(key=lambda x: x[1]['r2'], reverse=True)
            
            summary += "  RELATIONSHIP MODELS (ranked by R²):\n"
            for i, (rel_name, rel_data) in enumerate(sorted_rels, 1):
                marker = "★" if i == 1 else "  "
                summary += f"  {marker} {rel_name.capitalize()}: {rel_data['equation']} (R² = {rel_data['r2']:.4f})\n"
            
            # Add ratio if available
            if 'ratio' in relationships:
                rel = relationships['ratio']
                summary += f"    Ratio: {rel['equation']}\n"
            
            # Show best relationship
            if sorted_rels:
                best_rel_name, best_rel_data = sorted_rels[0]
                summary += f"\n  → BEST RELATIONSHIP: {best_rel_name.upper()}\n"
                summary += f"    {best_rel_data['equation']} (R² = {best_rel_data['r2']:.4f})\n"
                
        self.summary_text.insert(1.0, summary)
        
    def update_plots(self):
        """Update all plots"""
        self.plot_individual_fits()
        self.plot_comparison()
        
    def plot_individual_fits(self):
        """Plot individual curve fits"""
        self.fig_individual.clear()
        
        if self.water_fit_results or self.air_fit_results:
            # Create subplots
            if self.water_fit_results and self.air_fit_results:
                ax1 = self.fig_individual.add_subplot(2, 1, 1)
                ax2 = self.fig_individual.add_subplot(2, 1, 2)
            else:
                ax1 = self.fig_individual.add_subplot(1, 1, 1)
                ax2 = None
                
            # Plot water data
            if self.water_fit_results:
                self.plot_fits(ax1, self.water_fit_results, "Water Data")
                
            # Plot air data
            if self.air_fit_results and ax2:
                self.plot_fits(ax2, self.air_fit_results, "Air Data")
                
        self.fig_individual.tight_layout()
        self.canvas_individual.draw()
        
    def plot_fits(self, ax, fit_results, title):
        """Plot fits for a single dataset"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot original data
        first_result = list(fit_results.values())[0]
        x_data = first_result['x_data']
        y_data = first_result['y_data']
        
        ax.scatter(x_data, y_data, alpha=0.6, s=30, color='black', label='Original data')
        
        # Plot fits
        x_smooth = np.linspace(np.min(x_data), np.max(x_data), 200)
        
        for i, (model_name, results) in enumerate(fit_results.items()):
            color = colors[i % len(colors)]
            try:
                y_smooth = results['function'](x_smooth)
                label = f"{model_name} (R² = {results['r2']:.3f})"
                ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=label)
            except:
                continue
                
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_comparison(self):
        """Plot comparison and relationship"""
        self.fig_comparison.clear()
        
        if self.relationship_results:
            # Create subplots
            ax1 = self.fig_comparison.add_subplot(2, 2, 1)
            ax2 = self.fig_comparison.add_subplot(2, 2, 2)
            ax3 = self.fig_comparison.add_subplot(2, 2, 3)
            ax4 = self.fig_comparison.add_subplot(2, 2, 4)
            
            rel = self.relationship_results
            
            # Plot 1: Overlay comparison
            ax1.plot(rel['comparison_x'], rel['water_y'], 'b-', linewidth=2, label='Water (fitted)')
            ax1.plot(rel['comparison_x'], rel['air_y'], 'r-', linewidth=2, label='Air (fitted)')
            ax1.set_title('Comparación de Curvas Ajustadas')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Y_water vs Y_air (inverse relationship)
            ax2.scatter(rel['air_y'], rel['water_y'], alpha=0.6, s=20)
            if 'linear' in rel['relationships']:
                linear_rel = rel['relationships']['linear']
                # Use the inverse parameters: Y_water = a_inv * Y_air + b_inv
                y_fit = np.polyval(linear_rel['params'], rel['air_y'])
                ax2.plot(rel['air_y'], y_fit, 'r-', linewidth=2, 
                        label=f"R² = {linear_rel['r2']:.3f}")
                ax2.legend()
            ax2.set_title('Y_water vs Y_air Relationship')
            ax2.set_xlabel('Y_air')
            ax2.set_ylabel('Y_water')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Inverse Ratio (Y_water / Y_air)
            ratio_inv = rel['water_y'] / (rel['air_y'] + 1e-10)
            ax3.plot(rel['comparison_x'], ratio_inv, 'g-', linewidth=2)
            if 'ratio' in rel['relationships']:
                mean_ratio = rel['relationships']['ratio']['mean']
                ax3.axhline(y=mean_ratio, color='orange', linestyle='--', 
                           label=f'Average = {mean_ratio:.3f}')
                ax3.legend()
            ax3.set_title('Ratio Y_water / Y_air')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Ratio')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Inverse Difference (Y_water - Y_air)
            difference = rel['water_y'] - rel['air_y']
            ax4.plot(rel['comparison_x'], difference, 'm-', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('Difference (Y_water - Y_air)')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Diferencia')
            ax4.grid(True, alpha=0.3)
            
        else:
            # Plot original data if available
            if self.water_fit_results and self.air_fit_results:
                ax = self.fig_comparison.add_subplot(1, 1, 1)
                
                water_first = list(self.water_fit_results.values())[0]
                air_first = list(self.air_fit_results.values())[0]
                
                ax.scatter(water_first['x_data'], water_first['y_data'], 
                          alpha=0.6, s=30, color='blue', label='Water Data')
                ax.scatter(air_first['x_data'], air_first['y_data'], 
                          alpha=0.6, s=30, color='red', label='Air Data')
                
                ax.set_title('Original Data')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        self.fig_comparison.tight_layout()
        self.canvas_comparison.draw()
        
    def view_linear_relationships(self):
        """Display a graph showing ONLY the LINEAR relationships found for pressure drop in water and air for each sample"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments available. Please save some experiments first.")
            return
            
        # Filter experiments that have linear relationships
        linear_experiments = []
        for exp in self.experiments:
            if (exp.get('water_fit_results') and 'linear' in exp['water_fit_results'] and
                exp.get('air_fit_results') and 'linear' in exp['air_fit_results']):
                linear_experiments.append(exp)
        
        if not linear_experiments:
            messagebox.showwarning("No Linear Data", "No experiments with linear relationships found.")
            return
            
        # Create a new window for linear relationships visualization
        linear_window = tk.Toplevel(self.root)
        linear_window.title("Linear Relationships Only - Water and Air Pressure Drop")
        linear_window.geometry("1400x900")
        linear_window.transient(self.root)
        
        # Create notebook for different views
        notebook = ttk.Notebook(linear_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Individual Linear Fits
        individual_frame = ttk.Frame(notebook)
        notebook.add(individual_frame, text="Individual Linear Fits")
        
        # Tab 2: Linear Relationships Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Linear Relationships Summary")
        
        # Tab 3: Combined Linear Plot
        combined_frame = ttk.Frame(notebook)
        notebook.add(combined_frame, text="Combined Linear Analysis")
        
        # Create figures for each tab
        self.create_individual_linear_plots(individual_frame, linear_experiments)
        self.create_linear_summary_table(summary_frame, linear_experiments)
        self.create_combined_linear_plot(combined_frame, linear_experiments)
        
        # Add control buttons
        control_frame = ttk.Frame(linear_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Export Linear Results", 
                  command=lambda: self.export_linear_results(linear_experiments)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Calculate Average Relationship", 
                  command=self.calculate_average_linear_relationship).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", 
                  command=linear_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def create_individual_linear_plots(self, parent, linear_experiments):
        """Create individual linear plots for each experiment"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Calculate subplot grid
        n_experiments = len(linear_experiments)
        cols = min(3, n_experiments)
        rows = (n_experiments + cols - 1) // cols
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, exp in enumerate(linear_experiments):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Get linear fit data
            water_linear = exp['water_fit_results']['linear']
            air_linear = exp['air_fit_results']['linear']
            
            water_x = water_linear['x_data']
            water_y = water_linear['y_data']
            air_x = air_linear['x_data']
            air_y = air_linear['y_data']
            
            # Plot original data points
            ax.scatter(water_x, water_y, alpha=0.6, s=30, color='blue', label=f'Water Data (R²={water_linear["r2"]:.3f})')
            ax.scatter(air_x, air_y, alpha=0.6, s=30, color='red', label=f'Air Data (R²={air_linear["r2"]:.3f})')
            
            # Plot linear fits
            x_range_water = np.linspace(np.min(water_x), np.max(water_x), 100)
            x_range_air = np.linspace(np.min(air_x), np.max(air_x), 100)
            
            y_fit_water = water_linear['function'](x_range_water)
            y_fit_air = air_linear['function'](x_range_air)
            
            ax.plot(x_range_water, y_fit_water, 'b-', linewidth=2, alpha=0.8)
            ax.plot(x_range_air, y_fit_air, 'r-', linewidth=2, alpha=0.8)
            
            ax.set_title(f"{exp['name']}\nLinear Fits Only", fontsize=10)
            ax.set_xlabel('Flow/Velocity Parameter')
            ax.set_ylabel('Pressure Drop')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas.draw()
    
    def create_linear_summary_table(self, parent, linear_experiments):
        """Create summary table of linear relationships"""
        # Create frame with scrollbar
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview
        columns = ('Experiment', 'Water Linear Eq', 'Water R²', 'Air Linear Eq', 'Air R²', 
                  'Relationship Eq', 'Relationship R²', 'Ratio Factor')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        tree.column('Experiment', width=120)
        tree.column('Water Linear Eq', width=150)
        tree.column('Water R²', width=80)
        tree.column('Air Linear Eq', width=150)
        tree.column('Air R²', width=80)
        tree.column('Relationship Eq', width=180)
        tree.column('Relationship R²', width=100)
        tree.column('Ratio Factor', width=100)
        
        for col in columns:
            tree.heading(col, text=col)
        
        # Populate with data
        for exp in linear_experiments:
            water_linear = exp['water_fit_results']['linear']
            air_linear = exp['air_fit_results']['linear']
            
            # Format linear equations
            water_params = water_linear['params']
            air_params = air_linear['params']
            water_eq = f"y = {water_params[0]:.4f}x + {water_params[1]:.4f}"
            air_eq = f"y = {air_params[0]:.4f}x + {air_params[1]:.4f}"
            
            # Get relationship data if available
            relationship_eq = "N/A"
            relationship_r2 = "N/A"
            ratio_factor = "N/A"
            
            if (exp.get('relationship_results') and 
                'relationships' in exp['relationship_results'] and
                'linear' in exp['relationship_results']['relationships']):
                
                linear_rel = exp['relationship_results']['relationships']['linear']
                relationship_eq = linear_rel['equation']
                relationship_r2 = f"{linear_rel['r2']:.4f}"
                
                # Calculate ratio factor (slope relationship)
                if 'ratio' in exp['relationship_results']['relationships']:
                    ratio = exp['relationship_results']['relationships']['ratio']
                    ratio_factor = f"{ratio['mean']:.4f} ± {ratio['std']:.4f}"
            
            tree.insert('', tk.END, values=(
                exp['name'],
                water_eq,
                f"{water_linear['r2']:.4f}",
                air_eq,
                f"{air_linear['r2']:.4f}",
                relationship_eq,
                relationship_r2,
                ratio_factor
            ))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add statistics summary
        stats_frame = ttk.LabelFrame(parent, text="Linear Relationships Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        stats_text = tk.Text(stats_frame, height=8, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Calculate statistics
        water_r2_values = [exp['water_fit_results']['linear']['r2'] for exp in linear_experiments]
        air_r2_values = [exp['air_fit_results']['linear']['r2'] for exp in linear_experiments]
        
        stats_content = f"LINEAR RELATIONSHIPS STATISTICS ({len(linear_experiments)} experiments)\n"
        stats_content += "=" * 60 + "\n\n"
        stats_content += f"Water Linear Fits:\n"
        stats_content += f"  Average R²: {np.mean(water_r2_values):.4f}\n"
        stats_content += f"  Min R²: {np.min(water_r2_values):.4f}\n"
        stats_content += f"  Max R²: {np.max(water_r2_values):.4f}\n"
        stats_content += f"  Std Dev R²: {np.std(water_r2_values):.4f}\n\n"
        
        stats_content += f"Air Linear Fits:\n"
        stats_content += f"  Average R²: {np.mean(air_r2_values):.4f}\n"
        stats_content += f"  Min R²: {np.min(air_r2_values):.4f}\n"
        stats_content += f"  Max R²: {np.max(air_r2_values):.4f}\n"
        stats_content += f"  Std Dev R²: {np.std(air_r2_values):.4f}\n\n"
        
        # Quality assessment
        good_water_fits = sum(1 for r2 in water_r2_values if r2 > 0.8)
        good_air_fits = sum(1 for r2 in air_r2_values if r2 > 0.8)
        
        stats_content += f"Quality Assessment (R² > 0.8):\n"
        stats_content += f"  Good water fits: {good_water_fits}/{len(linear_experiments)} ({100*good_water_fits/len(linear_experiments):.1f}%)\n"
        stats_content += f"  Good air fits: {good_air_fits}/{len(linear_experiments)} ({100*good_air_fits/len(linear_experiments):.1f}%)\n"
        
        stats_text.insert(tk.END, stats_content)
        stats_text.config(state=tk.DISABLED)
    
    def create_combined_linear_plot(self, parent, linear_experiments):
        """Create combined plot showing all linear relationships"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(linear_experiments)))
        
        # Plot 1: All water linear fits
        for i, exp in enumerate(linear_experiments):
            water_linear = exp['water_fit_results']['linear']
            x_data = water_linear['x_data']
            y_data = water_linear['y_data']
            
            # Normalize x-range for comparison
            x_norm = np.linspace(0, 1, 100)
            x_actual = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_fit = water_linear['function'](x_actual)
            
            ax1.plot(x_actual, y_fit, color=colors[i], linewidth=2, 
                    label=f"{exp['name']} (R²={water_linear['r2']:.3f})", alpha=0.8)
        
        ax1.set_title('Water Linear Fits - All Experiments')
        ax1.set_xlabel('Flow Parameter')
        ax1.set_ylabel('Water Pressure Drop')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All air linear fits
        for i, exp in enumerate(linear_experiments):
            air_linear = exp['air_fit_results']['linear']
            x_data = air_linear['x_data']
            y_data = air_linear['y_data']
            
            x_actual = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_fit = air_linear['function'](x_actual)
            
            ax2.plot(x_actual, y_fit, color=colors[i], linewidth=2, 
                    label=f"{exp['name']} (R²={air_linear['r2']:.3f})", alpha=0.8)
        
        ax2.set_title('Air Linear Fits - All Experiments')
        ax2.set_xlabel('Flow Parameter')
        ax2.set_ylabel('Air Pressure Drop')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² comparison
        exp_names = [exp['name'] for exp in linear_experiments]
        water_r2 = [exp['water_fit_results']['linear']['r2'] for exp in linear_experiments]
        air_r2 = [exp['air_fit_results']['linear']['r2'] for exp in linear_experiments]
        
        x_pos = np.arange(len(exp_names))
        width = 0.35
        
        ax3.bar(x_pos - width/2, water_r2, width, label='Water R²', alpha=0.8, color='blue')
        ax3.bar(x_pos + width/2, air_r2, width, label='Air R²', alpha=0.8, color='red')
        
        ax3.set_title('Linear Fit Quality Comparison')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('R² Value')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(exp_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Slope comparison (pressure drop relationship)
        water_slopes = [exp['water_fit_results']['linear']['params'][0] for exp in linear_experiments]
        air_slopes = [exp['air_fit_results']['linear']['params'][0] for exp in linear_experiments]
        
        ax4.scatter(water_slopes, air_slopes, s=100, alpha=0.7, c=range(len(linear_experiments)), cmap='tab10')
        
        for i, exp in enumerate(linear_experiments):
            ax4.annotate(exp['name'], (water_slopes[i], air_slopes[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        if len(water_slopes) > 1:
            z = np.polyfit(water_slopes, air_slopes, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(water_slopes), max(water_slopes), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            correlation = np.corrcoef(water_slopes, air_slopes)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        ax4.set_title('Slope Relationship: Water vs Air')
        ax4.set_xlabel('Water Slope (dP/dFlow)')
        ax4.set_ylabel('Air Slope (dP/dFlow)')
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas.draw()
    
    def calculate_average_linear_relationship(self):
        """Calculate average linear relationship from all experiments for estimating water pressure drop"""
        # Get all experiments with linear relationships
        linear_experiments = []
        for exp in self.experiments:
            if (exp.get('water_fit_results') and 'linear' in exp['water_fit_results'] and
                exp.get('air_fit_results') and 'linear' in exp['air_fit_results'] and
                exp.get('relationship_results') and 'relationships' in exp['relationship_results'] and
                'linear' in exp['relationship_results']['relationships']):
                linear_experiments.append(exp)
        
        if len(linear_experiments) < 2:
            messagebox.showwarning("Insufficient Data", 
                                  "Need at least 2 experiments with linear relationships to calculate average.")
            return
        
        # Extract relationship parameters
        relationship_params = []
        relationship_r2_values = []
        ratio_means = []
        
        for exp in linear_experiments:
            linear_rel = exp['relationship_results']['relationships']['linear']
            relationship_params.append(linear_rel['params'])
            relationship_r2_values.append(linear_rel['r2'])
            
            if 'ratio' in exp['relationship_results']['relationships']:
                ratio_means.append(exp['relationship_results']['relationships']['ratio']['mean'])
        
        # Calculate weighted average (weighted by R²)
        weights = np.array(relationship_r2_values)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Average parameters
        avg_params = np.average(relationship_params, axis=0, weights=weights)
        avg_r2 = np.mean(relationship_r2_values)
        
        # Average ratio if available
        avg_ratio = np.mean(ratio_means) if ratio_means else None
        ratio_std = np.std(ratio_means) if ratio_means else None
        
        # Create average relationship function
        def avg_relationship_func(y_air):
            """Average relationship function to estimate water pressure drop from air pressure drop"""
            return avg_params[0] * y_air + avg_params[1]
        
        # Store the average relationship
        self.average_relationship = {
            'params': avg_params,
            'r2_avg': avg_r2,
            'function': avg_relationship_func,
            'equation': f"Y_water = {avg_params[0]:.4f} * Y_air + {avg_params[1]:.4f}",
            'n_experiments': len(linear_experiments),
            'weights': weights,
            'individual_r2': relationship_r2_values,
            'ratio_mean': avg_ratio,
            'ratio_std': ratio_std
        }
        
        # Create results window
        self.show_average_relationship_results(linear_experiments)
    
    def show_average_relationship_results(self, linear_experiments):
        """Show the results of the average linear relationship calculation"""
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Average Linear Relationship for Water Pressure Drop Estimation")
        results_window.geometry("1200x800")
        results_window.transient(self.root)
        
        # Create notebook
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Results Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Average Relationship Results")
        
        # Tab 2: Validation Plot
        validation_frame = ttk.Frame(notebook)
        notebook.add(validation_frame, text="Validation & Prediction")
        
        # Tab 3: Usage Guide
        guide_frame = ttk.Frame(notebook)
        notebook.add(guide_frame, text="Usage Guide")
        
        # Create results summary
        self.create_average_relationship_summary(summary_frame, linear_experiments)
        self.create_validation_plot(validation_frame, linear_experiments)
        self.create_usage_guide(guide_frame)
        
        # Control buttons
        control_frame = ttk.Frame(results_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Export Average Relationship", 
                  command=self.export_average_relationship).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Test Prediction", 
                  command=self.test_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", 
                  command=results_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def create_average_relationship_summary(self, parent, linear_experiments):
        """Create summary of the average relationship"""
        # Main frame with text widget
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=('Courier', 11))
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create summary content
        avg_rel = self.average_relationship
        
        content = "AVERAGE LINEAR RELATIONSHIP FOR WATER PRESSURE DROP ESTIMATION\n"
        content += "=" * 80 + "\n\n"
        
        content += f"📊 MAIN EQUATION (for estimation):\n"
        content += f"   {avg_rel['equation']}\n\n"
        
        content += f"📈 STATISTICAL SUMMARY:\n"
        content += f"   • Number of experiments used: {avg_rel['n_experiments']}\n"
        content += f"   • Average R² of relationships: {avg_rel['r2_avg']:.4f}\n"
        
        if avg_rel['ratio_mean'] is not None:
            content += f"   • Average ratio (Y_water/Y_air): {avg_rel['ratio_mean']:.4f} ± {avg_rel['ratio_std']:.4f}\n"
        
        content += f"\n📋 INDIVIDUAL EXPERIMENT CONTRIBUTIONS:\n"
        content += "   Exp.  Weight   R²     Equation\n"
        content += "   " + "-" * 50 + "\n"
        
        for i, exp in enumerate(linear_experiments):
            weight = avg_rel['weights'][i]
            r2 = avg_rel['individual_r2'][i]
            rel = exp['relationship_results']['relationships']['linear']
            params = rel['params']
            eq_short = f"Y = {params[0]:.3f}*X + {params[1]:.3f}"
            content += f"   {i+1:2d}.   {weight:.3f}   {r2:.3f}   {eq_short}\n"
        
        content += f"\n🎯 ACCURACY ASSESSMENT:\n"
        r2_values = avg_rel['individual_r2']
        content += f"   • Best individual R²: {max(r2_values):.4f}\n"
        content += f"   • Worst individual R²: {min(r2_values):.4f}\n"
        content += f"   • Standard deviation of R²: {np.std(r2_values):.4f}\n"
        
        # Quality rating
        if avg_rel['r2_avg'] > 0.9:
            quality = "EXCELLENT"
        elif avg_rel['r2_avg'] > 0.8:
            quality = "GOOD"
        elif avg_rel['r2_avg'] > 0.7:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
        
        content += f"   • Overall quality rating: {quality}\n"
        
        content += f"\n🔧 USAGE INSTRUCTIONS:\n"
        content += f"   1. Measure air pressure drop (Y_air) for your system\n"
        content += f"   2. Apply the equation: Y_water = {avg_rel['params'][0]:.4f} * Y_air + {avg_rel['params'][1]:.4f}\n"
        content += f"   3. The result gives you the estimated water pressure drop\n"
        content += f"   4. Expected accuracy: ±{(1-avg_rel['r2_avg'])*100:.1f}% based on R² = {avg_rel['r2_avg']:.3f}\n"
        
        content += f"\n⚠️  IMPORTANT NOTES:\n"
        content += f"   • This relationship is valid for similar operating conditions\n"
        content += f"   • Extrapolation beyond the tested range may reduce accuracy\n"
        content += f"   • Regular validation with actual measurements is recommended\n"
        
        if avg_rel['ratio_mean'] is not None:
            content += f"\n📊 ALTERNATIVE SIMPLE ESTIMATION:\n"
            content += f"   Y_water ≈ {avg_rel['ratio_mean']:.4f} * Y_air\n"
            content += f"   (Simple ratio method, accuracy may vary ±{avg_rel['ratio_std']:.4f})\n"
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def create_validation_plot(self, parent, linear_experiments):
        """Create validation plot showing prediction accuracy"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        avg_rel = self.average_relationship
        colors = plt.cm.tab10(np.linspace(0, 1, len(linear_experiments)))
        
        # Collect validation data
        all_air_y = []
        all_water_y = []
        all_predicted_y = []
        
        # Plot 1: Individual relationships vs average
        for i, exp in enumerate(linear_experiments):
            rel = exp['relationship_results']['relationships']['linear']
            
            # Get data range for this experiment
            if exp['relationship_results']:
                air_y = exp['relationship_results']['air_y']
                water_y = exp['relationship_results']['water_y']
                
                all_air_y.extend(air_y)
                all_water_y.extend(water_y)
                
                # Plot individual relationship
                ax1.scatter(air_y, water_y, alpha=0.6, s=20, color=colors[i], 
                          label=f"{exp['name']} (R²={rel['r2']:.3f})")
                
                # Individual fit line
                air_range = np.linspace(np.min(air_y), np.max(air_y), 50)
                water_individual = rel['params'][0] * air_range + rel['params'][1]
                ax1.plot(air_range, water_individual, color=colors[i], linewidth=1, alpha=0.8)
                
                # Predictions using average relationship
                predicted_y = avg_rel['function'](air_y)
                all_predicted_y.extend(predicted_y)
        
        # Plot average relationship line
        if all_air_y:
            air_overall_range = np.linspace(min(all_air_y), max(all_air_y), 100)
            water_avg_pred = avg_rel['function'](air_overall_range)
            ax1.plot(air_overall_range, water_avg_pred, 'k-', linewidth=3, alpha=0.9, 
                    label=f'Average Relationship (R²_avg={avg_rel["r2_avg"]:.3f})')
        
        ax1.set_title('Individual vs Average Relationship')
        ax1.set_xlabel('Air Pressure Drop')
        ax1.set_ylabel('Water Pressure Drop')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction accuracy
        if all_water_y and all_predicted_y:
            ax2.scatter(all_water_y, all_predicted_y, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(min(all_water_y), min(all_predicted_y))
            max_val = max(max(all_water_y), max(all_predicted_y))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
            
            # Calculate prediction R²
            pred_r2 = r2_score(all_water_y, all_predicted_y)
            
            ax2.set_title(f'Prediction Accuracy (R² = {pred_r2:.4f})')
            ax2.set_xlabel('Actual Water Pressure Drop')
            ax2.set_ylabel('Predicted Water Pressure Drop')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            residuals = np.array(all_predicted_y) - np.array(all_water_y)
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            
            ax2.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
        
        # Plot 3: Residuals
        if all_water_y and all_predicted_y:
            residuals = np.array(all_predicted_y) - np.array(all_water_y)
            ax3.scatter(all_predicted_y, residuals, alpha=0.6, s=30)
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            
            ax3.set_title('Residuals Plot')
            ax3.set_xlabel('Predicted Water Pressure Drop')
            ax3.set_ylabel('Residuals (Predicted - Actual)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        if all_water_y and all_predicted_y:
            residuals = np.array(all_predicted_y) - np.array(all_water_y)
            relative_errors = 100 * residuals / (np.array(all_water_y) + 1e-10)
            
            ax4.hist(relative_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8)
            ax4.axvline(x=np.mean(relative_errors), color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(relative_errors):.2f}%')
            
            ax4.set_title('Prediction Error Distribution')
            ax4.set_xlabel('Relative Error (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas.draw()
    
    def create_usage_guide(self, parent):
        """Create usage guide for the average relationship"""
        # Create text widget with instructions
        text_widget = tk.Text(parent, wrap=tk.WORD, font=('Arial', 11))
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        guide_content = """HOW TO USE THE AVERAGE LINEAR RELATIONSHIP FOR WATER PRESSURE DROP ESTIMATION

🎯 PURPOSE:
This tool provides you with a calibrated equation to estimate water pressure drop based on air pressure drop measurements. This is useful when:
• Water testing is difficult or expensive
• You need quick pressure drop estimates
• You want to validate water system designs using air tests

📐 THE EQUATION:
Your calibrated equation is: """ + self.average_relationship['equation'] + """

🔧 STEP-BY-STEP USAGE:

1. MEASURE AIR PRESSURE DROP (Y_air):
   • Set up your system with air as the working fluid
   • Measure pressure drop at various flow rates
   • Record the pressure drop value you want to convert

2. APPLY THE EQUATION:
   • Substitute your measured Y_air value into the equation
   • Calculate: Y_water = """ + f"{self.average_relationship['params'][0]:.4f}" + """ * Y_air + """ + f"{self.average_relationship['params'][1]:.4f}" + """
   • The result is your estimated water pressure drop

3. EXAMPLE CALCULATION:
   If you measure an air pressure drop of 100 Pa:
   Y_water = """ + f"{self.average_relationship['params'][0]:.4f}" + """ * 100 + """ + f"{self.average_relationship['params'][1]:.4f}" + """
   Y_water = """ + f"{self.average_relationship['params'][0] * 100 + self.average_relationship['params'][1]:.2f}" + """ Pa

⚠️ IMPORTANT CONSIDERATIONS:

• OPERATING CONDITIONS: This relationship is valid for similar conditions to your calibration experiments
• FLOW RANGE: Best accuracy within the flow range used for calibration
• SYSTEM SIMILARITY: Most accurate for geometrically similar systems
• VALIDATION: Periodically validate with actual water measurements

📊 ACCURACY EXPECTATIONS:
• Expected accuracy: ±""" + f"{(1-self.average_relationship['r2_avg'])*100:.1f}" + """% based on R² = """ + f"{self.average_relationship['r2_avg']:.3f}" + """
• Based on """ + str(self.average_relationship['n_experiments']) + """ calibration experiments

🔬 ALTERNATIVE SIMPLE METHOD:
If you prefer a simple ratio approach:"""

        if self.average_relationship['ratio_mean'] is not None:
            guide_content += f"\nY_water ≈ {self.average_relationship['ratio_mean']:.4f} * Y_air"
            guide_content += f"\n(Accuracy may vary ±{self.average_relationship['ratio_std']:.4f})"
        
        guide_content += """

🎯 WHEN TO RECALIBRATE:
• When system geometry changes significantly
• When operating conditions change substantially
• When accuracy validation shows degraded performance
• After adding new experimental data

💡 TIPS FOR BEST RESULTS:
• Use similar flow rates between air and water tests
• Maintain similar Reynolds numbers when possible
• Account for temperature effects if significant
• Keep detailed records of operating conditions

📁 EXPORT OPTIONS:
• Use "Export Average Relationship" to save the equation and parameters
• Save this analysis for future reference
• Include the equation in your design documentation
"""
        
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED)
    
    def export_linear_results(self, linear_experiments):
        """Export linear relationships analysis"""
        file_path = filedialog.asksaveasfilename(
            title="Export Linear Relationships Analysis",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    for exp in linear_experiments:
                        water_linear = exp['water_fit_results']['linear']
                        air_linear = exp['air_fit_results']['linear']
                        
                        row = {
                            'Experiment_Name': exp['name'],
                            'Water_Linear_R2': water_linear['r2'],
                            'Water_Slope': water_linear['params'][0],
                            'Water_Intercept': water_linear['params'][1],
                            'Air_Linear_R2': air_linear['r2'],
                            'Air_Slope': air_linear['params'][0],
                            'Air_Intercept': air_linear['params'][1]
                        }
                        
                        if (exp.get('relationship_results') and 
                            'relationships' in exp['relationship_results'] and
                            'linear' in exp['relationship_results']['relationships']):
                            linear_rel = exp['relationship_results']['relationships']['linear']
                            row['Relationship_R2'] = linear_rel['r2']
                            row['Relationship_Equation'] = linear_rel['equation']
                        
                        summary_data.append(row)
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Linear_Analysis', index=False)
                    
                    # Statistics sheet
                    stats_data = {
                        'Metric': [
                            'Number of Experiments',
                            'Average Water R²',
                            'Average Air R²',
                            'Min Water R²',
                            'Max Water R²',
                            'Min Air R²',
                            'Max Air R²',
                            'Water R² Std Dev',
                            'Air R² Std Dev'
                        ],
                        'Value': [
                            len(linear_experiments),
                            np.mean([exp['water_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.mean([exp['air_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.min([exp['water_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.max([exp['water_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.min([exp['air_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.max([exp['air_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.std([exp['water_fit_results']['linear']['r2'] for exp in linear_experiments]),
                            np.std([exp['air_fit_results']['linear']['r2'] for exp in linear_experiments])
                        ]
                    }
                    
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                messagebox.showinfo("Success", f"Linear relationships analysis exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting linear analysis:\n{str(e)}")
    
    def export_average_relationship(self):
        """Export the average relationship equation and parameters"""
        if not hasattr(self, 'average_relationship'):
            messagebox.showwarning("No Data", "Please calculate the average relationship first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Average Relationship",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    # Export to Excel
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Main equation sheet
                        equation_data = {
                            'Parameter': [
                                'Equation',
                                'Slope (a)',
                                'Intercept (b)', 
                                'Average R²',
                                'Number of Experiments',
                                'Quality Rating'
                            ],
                            'Value': [
                                self.average_relationship['equation'],
                                self.average_relationship['params'][0],
                                self.average_relationship['params'][1],
                                self.average_relationship['r2_avg'],
                                self.average_relationship['n_experiments'],
                                'EXCELLENT' if self.average_relationship['r2_avg'] > 0.9 else 
                                'GOOD' if self.average_relationship['r2_avg'] > 0.8 else 
                                'ACCEPTABLE' if self.average_relationship['r2_avg'] > 0.7 else 'POOR'
                            ]
                        }
                        
                        equation_df = pd.DataFrame(equation_data)
                        equation_df.to_excel(writer, sheet_name='Average_Relationship', index=False)
                        
                        # Individual contributions
                        contrib_data = []
                        for i in range(self.average_relationship['n_experiments']):
                            contrib_data.append({
                                'Experiment_Index': i+1,
                                'Weight': self.average_relationship['weights'][i],
                                'Individual_R2': self.average_relationship['individual_r2'][i]
                            })
                        
                        contrib_df = pd.DataFrame(contrib_data)
                        contrib_df.to_excel(writer, sheet_name='Contributions', index=False)
                        
                        # Usage instructions
                        instructions = pd.DataFrame({
                            'Step': [
                                '1. Measure air pressure drop (Y_air)',
                                '2. Apply equation',
                                '3. Calculate result',
                                '4. Interpret result'
                            ],
                            'Description': [
                                'Measure pressure drop in your system using air',
                                f'Y_water = {self.average_relationship["params"][0]:.4f} * Y_air + {self.average_relationship["params"][1]:.4f}',
                                'Substitute Y_air value and calculate Y_water',
                                f'Expected accuracy: ±{(1-self.average_relationship["r2_avg"])*100:.1f}%'
                            ]
                        })
                        
                        instructions.to_excel(writer, sheet_name='Usage_Instructions', index=False)
                    
                else:
                    # Export to text file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("AVERAGE LINEAR RELATIONSHIP FOR WATER PRESSURE DROP ESTIMATION\n")
                        f.write("=" * 70 + "\n\n")
                        f.write(f"EQUATION: {self.average_relationship['equation']}\n\n")
                        f.write(f"Parameters:\n")
                        f.write(f"  Slope (a): {self.average_relationship['params'][0]:.6f}\n")
                        f.write(f"  Intercept (b): {self.average_relationship['params'][1]:.6f}\n")
                        f.write(f"  Average R²: {self.average_relationship['r2_avg']:.4f}\n")
                        f.write(f"  Based on {self.average_relationship['n_experiments']} experiments\n\n")
                        f.write("USAGE:\n")
                        f.write("1. Measure air pressure drop (Y_air)\n")
                        f.write(f"2. Calculate: Y_water = {self.average_relationship['params'][0]:.4f} * Y_air + {self.average_relationship['params'][1]:.4f}\n")
                        f.write(f"3. Expected accuracy: ±{(1-self.average_relationship['r2_avg'])*100:.1f}%\n")
                
                messagebox.showinfo("Success", f"Average relationship exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting average relationship:\n{str(e)}")
    
    def test_prediction(self):
        """Test the prediction with user-provided air pressure drop value"""
        if not hasattr(self, 'average_relationship'):
            messagebox.showwarning("No Data", "Please calculate the average relationship first.")
            return
            
        # Ask user for air pressure drop value
        air_value = tk.simpledialog.askfloat(
            "Test Prediction",
            "Enter air pressure drop value to predict water pressure drop:",
            minvalue=0.0
        )
        
        if air_value is not None:
            # Calculate prediction
            predicted_water = self.average_relationship['function'](air_value)
            
            # Show results
            result_msg = f"INPUT: Air pressure drop = {air_value}\n\n"
            result_msg += f"PREDICTION: Water pressure drop = {predicted_water:.4f}\n\n"
            result_msg += f"EQUATION USED: {self.average_relationship['equation']}\n\n"
            result_msg += f"CALCULATION:\n"
            result_msg += f"Y_water = {self.average_relationship['params'][0]:.4f} × {air_value} + {self.average_relationship['params'][1]:.4f}\n"
            result_msg += f"Y_water = {predicted_water:.4f}\n\n"
            result_msg += f"ACCURACY: Expected ±{(1-self.average_relationship['r2_avg'])*100:.1f}% based on R² = {self.average_relationship['r2_avg']:.3f}"
            
            messagebox.showinfo("Prediction Result", result_msg)
        
    def view_polynomial2_relationships(self):
        """Display a graph showing ONLY the POLYNOMIAL DEGREE 2 relationships found for pressure drop in water and air for each sample"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments available. Please save some experiments first.")
            return
            
        # Filter experiments that have polynomial degree 2 relationships
        poly2_experiments = []
        for exp in self.experiments:
            if (exp.get('water_fit_results') and 'poly2' in exp['water_fit_results'] and
                exp.get('air_fit_results') and 'poly2' in exp['air_fit_results']):
                poly2_experiments.append(exp)
        
        if not poly2_experiments:
            messagebox.showwarning("No Polynomial Data", "No experiments with polynomial degree 2 relationships found.")
            return
            
        # Create polynomial analysis window
        self.create_polynomial_analysis_window(poly2_experiments, "Polynomial Degree 2", "poly2")
    
    def view_polynomial3_relationships(self):
        """Display a graph showing ONLY the POLYNOMIAL DEGREE 3 relationships found for pressure drop in water and air for each sample"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments available. Please save some experiments first.")
            return
            
        # Filter experiments that have polynomial degree 3 relationships
        poly3_experiments = []
        for exp in self.experiments:
            if (exp.get('water_fit_results') and 'poly3' in exp['water_fit_results'] and
                exp.get('air_fit_results') and 'poly3' in exp['air_fit_results']):
                poly3_experiments.append(exp)
        
        if not poly3_experiments:
            messagebox.showwarning("No Polynomial Data", "No experiments with polynomial degree 3 relationships found.")
            return
            
        # Create polynomial analysis window
        self.create_polynomial_analysis_window(poly3_experiments, "Polynomial Degree 3", "poly3")
    
    def create_polynomial_analysis_window(self, poly_experiments, title, poly_type):
        """Create analysis window for polynomial relationships"""
        # Create a new window for polynomial relationships visualization
        poly_window = tk.Toplevel(self.root)
        poly_window.title(f"{title} Relationships Only - Water and Air Pressure Drop")
        poly_window.geometry("1400x900")
        poly_window.transient(self.root)
        
        # Create notebook for different views
        notebook = ttk.Notebook(poly_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Individual Polynomial Fits
        individual_frame = ttk.Frame(notebook)
        notebook.add(individual_frame, text=f"Individual {title} Fits")
        
        # Tab 2: Polynomial Relationships Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text=f"{title} Relationships Summary")
        
        # Tab 3: Combined Polynomial Plot
        combined_frame = ttk.Frame(notebook)
        notebook.add(combined_frame, text=f"Combined {title} Analysis")
        
        # Tab 4: Linear Relationship Between Polynomials
        linear_rel_frame = ttk.Frame(notebook)
        notebook.add(linear_rel_frame, text="Linear Relationship Analysis")
        
        # Create figures for each tab
        self.create_individual_polynomial_plots(individual_frame, poly_experiments, title, poly_type)
        self.create_polynomial_summary_table(summary_frame, poly_experiments, title, poly_type)
        self.create_combined_polynomial_plot(combined_frame, poly_experiments, title, poly_type)
        self.create_polynomial_linear_relationship_plot(linear_rel_frame, poly_experiments, title, poly_type)
        
        # Add control buttons
        control_frame = ttk.Frame(poly_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text=f"Export {title} Results", 
                  command=lambda: self.export_polynomial_results(poly_experiments, title, poly_type)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text=f"Calculate Average {title} Relationship", 
                  command=lambda: self.calculate_average_polynomial_relationship(poly_experiments, title, poly_type)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", 
                  command=poly_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def create_individual_polynomial_plots(self, parent, poly_experiments, title, poly_type):
        """Create individual polynomial plots for each experiment"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Calculate subplot grid
        n_experiments = len(poly_experiments)
        cols = min(3, n_experiments)
        rows = (n_experiments + cols - 1) // cols
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, exp in enumerate(poly_experiments):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Get polynomial fit data
            water_poly = exp['water_fit_results'][poly_type]
            air_poly = exp['air_fit_results'][poly_type]
            
            water_x = water_poly['x_data']
            water_y = water_poly['y_data']
            air_x = air_poly['x_data']
            air_y = air_poly['y_data']
            
            # Plot original data points
            ax.scatter(water_x, water_y, alpha=0.6, s=30, color='blue', label=f'Water Data (R²={water_poly["r2"]:.3f})')
            ax.scatter(air_x, air_y, alpha=0.6, s=30, color='red', label=f'Air Data (R²={air_poly["r2"]:.3f})')
            
            # Plot polynomial fits
            x_range_water = np.linspace(np.min(water_x), np.max(water_x), 100)
            x_range_air = np.linspace(np.min(air_x), np.max(air_x), 100)
            
            y_fit_water = water_poly['function'](x_range_water)
            y_fit_air = air_poly['function'](x_range_air)
            
            ax.plot(x_range_water, y_fit_water, 'b-', linewidth=2, alpha=0.8)
            ax.plot(x_range_air, y_fit_air, 'r-', linewidth=2, alpha=0.8)
            
            ax.set_title(f"{exp['name']}\n{title} Fits Only", fontsize=10)
            ax.set_xlabel('Flow/Velocity Parameter')
            ax.set_ylabel('Pressure Drop')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas.draw()
    
    def create_polynomial_summary_table(self, parent, poly_experiments, title, poly_type):
        """Create summary table of polynomial relationships"""
        # Create frame with scrollbar
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview
        columns = ('Experiment', f'Water {title} R²', f'Air {title} R²', 
                  'Water Equation', 'Air Equation', 'Quality Assessment')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        tree.column('Experiment', width=120)
        tree.column(f'Water {title} R²', width=100)
        tree.column(f'Air {title} R²', width=100)
        tree.column('Water Equation', width=200)
        tree.column('Air Equation', width=200)
        tree.column('Quality Assessment', width=120)
        
        for col in columns:
            tree.heading(col, text=col)
        
        # Populate with data
        for exp in poly_experiments:
            water_poly = exp['water_fit_results'][poly_type]
            air_poly = exp['air_fit_results'][poly_type]
            
            # Format polynomial equations
            water_params = water_poly['params']
            air_params = air_poly['params']
            
            if poly_type == 'poly2':
                water_eq = f"y = {water_params[0]:.4f}x² + {water_params[1]:.4f}x + {water_params[2]:.4f}"
                air_eq = f"y = {air_params[0]:.4f}x² + {air_params[1]:.4f}x + {air_params[2]:.4f}"
            else:  # poly3
                water_eq = f"y = {water_params[0]:.4f}x³ + {water_params[1]:.4f}x² + {water_params[2]:.4f}x + {water_params[3]:.4f}"
                air_eq = f"y = {air_params[0]:.4f}x³ + {air_params[1]:.4f}x² + {air_params[2]:.4f}x + {air_params[3]:.4f}"
            
            # Quality assessment
            min_r2 = min(water_poly['r2'], air_poly['r2'])
            quality = 'EXCELLENT' if min_r2 > 0.9 else 'GOOD' if min_r2 > 0.8 else 'ACCEPTABLE' if min_r2 > 0.7 else 'POOR'
            
            tree.insert('', tk.END, values=(
                exp['name'],
                f"{water_poly['r2']:.4f}",
                f"{air_poly['r2']:.4f}",
                water_eq,
                air_eq,
                quality
            ))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add statistics summary
        stats_frame = ttk.LabelFrame(parent, text=f"{title} Relationships Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        stats_text = tk.Text(stats_frame, height=8, wrap=tk.WORD)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Calculate statistics
        water_r2_values = [exp['water_fit_results'][poly_type]['r2'] for exp in poly_experiments]
        air_r2_values = [exp['air_fit_results'][poly_type]['r2'] for exp in poly_experiments]
        
        stats_content = f"{title.upper()} RELATIONSHIPS STATISTICS ({len(poly_experiments)} experiments)\n"
        stats_content += "=" * 60 + "\n\n"
        stats_content += f"Water {title} Fits:\n"
        stats_content += f"  Average R²: {np.mean(water_r2_values):.4f}\n"
        stats_content += f"  Min R²: {np.min(water_r2_values):.4f}\n"
        stats_content += f"  Max R²: {np.max(water_r2_values):.4f}\n"
        stats_content += f"  Std Dev R²: {np.std(water_r2_values):.4f}\n\n"
        
        stats_content += f"Air {title} Fits:\n"
        stats_content += f"  Average R²: {np.mean(air_r2_values):.4f}\n"
        stats_content += f"  Min R²: {np.min(air_r2_values):.4f}\n"
        stats_content += f"  Max R²: {np.max(air_r2_values):.4f}\n"
        stats_content += f"  Std Dev R²: {np.std(air_r2_values):.4f}\n\n"
        
        # Quality assessment
        good_water_fits = sum(1 for r2 in water_r2_values if r2 > 0.8)
        good_air_fits = sum(1 for r2 in air_r2_values if r2 > 0.8)
        
        stats_content += f"Quality Assessment (R² > 0.8):\n"
        stats_content += f"  Good water fits: {good_water_fits}/{len(poly_experiments)} ({100*good_water_fits/len(poly_experiments):.1f}%)\n"
        stats_content += f"  Good air fits: {good_air_fits}/{len(poly_experiments)} ({100*good_air_fits/len(poly_experiments):.1f}%)\n"
        
        stats_text.insert(tk.END, stats_content)
        stats_text.config(state=tk.DISABLED)
    
    def create_combined_polynomial_plot(self, parent, poly_experiments, title, poly_type):
        """Create combined plot showing all polynomial relationships"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(poly_experiments)))
        
        # Plot 1: All water polynomial fits
        for i, exp in enumerate(poly_experiments):
            water_poly = exp['water_fit_results'][poly_type]
            x_data = water_poly['x_data']
            y_data = water_poly['y_data']
            
            x_actual = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_fit = water_poly['function'](x_actual)
            
            ax1.plot(x_actual, y_fit, color=colors[i], linewidth=2, 
                    label=f"{exp['name']} (R²={water_poly['r2']:.3f})", alpha=0.8)
        
        ax1.set_title(f'Water {title} Fits - All Experiments')
        ax1.set_xlabel('Flow Parameter')
        ax1.set_ylabel('Water Pressure Drop')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All air polynomial fits
        for i, exp in enumerate(poly_experiments):
            air_poly = exp['air_fit_results'][poly_type]
            x_data = air_poly['x_data']
            y_data = air_poly['y_data']
            
            x_actual = np.linspace(np.min(x_data), np.max(x_data), 100)
            y_fit = air_poly['function'](x_actual)
            
            ax2.plot(x_actual, y_fit, color=colors[i], linewidth=2, 
                    label=f"{exp['name']} (R²={air_poly['r2']:.3f})", alpha=0.8)
        
        ax2.set_title(f'Air {title} Fits - All Experiments')
        ax2.set_xlabel('Flow Parameter')
        ax2.set_ylabel('Air Pressure Drop')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² comparison
        exp_names = [exp['name'] for exp in poly_experiments]
        water_r2 = [exp['water_fit_results'][poly_type]['r2'] for exp in poly_experiments]
        air_r2 = [exp['air_fit_results'][poly_type]['r2'] for exp in poly_experiments]
        
        x_pos = np.arange(len(exp_names))
        width = 0.35
        
        ax3.bar(x_pos - width/2, water_r2, width, label='Water R²', alpha=0.8, color='blue')
        ax3.bar(x_pos + width/2, air_r2, width, label='Air R²', alpha=0.8, color='red')
        
        ax3.set_title(f'{title} Fit Quality Comparison')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('R² Value')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(exp_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Coefficient comparison
        if poly_type == 'poly2':
            # For degree 2: compare quadratic coefficients
            water_quad_coeff = [exp['water_fit_results'][poly_type]['params'][0] for exp in poly_experiments]
            air_quad_coeff = [exp['air_fit_results'][poly_type]['params'][0] for exp in poly_experiments]
            coeff_name = 'Quadratic Coefficient (a)'
        else:  # poly3
            # For degree 3: compare cubic coefficients
            water_quad_coeff = [exp['water_fit_results'][poly_type]['params'][0] for exp in poly_experiments]
            air_quad_coeff = [exp['air_fit_results'][poly_type]['params'][0] for exp in poly_experiments]
            coeff_name = 'Cubic Coefficient (a)'
        
        ax4.scatter(water_quad_coeff, air_quad_coeff, s=100, alpha=0.7, c=range(len(poly_experiments)), cmap='tab10')
        
        for i, exp in enumerate(poly_experiments):
            ax4.annotate(exp['name'], (water_quad_coeff[i], air_quad_coeff[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        if len(water_quad_coeff) > 1:
            z = np.polyfit(water_quad_coeff, air_quad_coeff, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(water_quad_coeff), max(water_quad_coeff), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            correlation = np.corrcoef(water_quad_coeff, air_quad_coeff)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        ax4.set_title(f'{coeff_name} Relationship')
        ax4.set_xlabel(f'Water {coeff_name}')
        ax4.set_ylabel(f'Air {coeff_name}')
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas.draw()
    
    def create_polynomial_linear_relationship_plot(self, parent, poly_experiments, title, poly_type):
        """Create plot showing linear relationship between polynomial curves"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(poly_experiments)))
        
        # Calculate linear relationships between polynomial curves
        linear_relationships = []
        
        for i, exp in enumerate(poly_experiments):
            water_poly = exp['water_fit_results'][poly_type]
            air_poly = exp['air_fit_results'][poly_type]
            
            # Generate comparison points
            x_min = max(np.min(water_poly['x_data']), np.min(air_poly['x_data']))
            x_max = min(np.max(water_poly['x_data']), np.max(air_poly['x_data']))
            
            x_comparison = np.linspace(x_min, x_max, 100)
            
            water_y = water_poly['function'](x_comparison)
            air_y = air_poly['function'](x_comparison)
            
            # Fit linear relationship: water_y = a * air_y + b
            try:
                # Remove any invalid values
                valid_mask = np.isfinite(water_y) & np.isfinite(air_y)
                if np.sum(valid_mask) > 10:  # Need enough points
                    water_y_valid = water_y[valid_mask]
                    air_y_valid = air_y[valid_mask]
                    
                    # Fit linear relationship
                    linear_params = np.polyfit(air_y_valid, water_y_valid, 1)
                    y_pred = np.polyval(linear_params, air_y_valid)
                    r2_linear = r2_score(water_y_valid, y_pred)
                    
                    linear_relationships.append({
                        'experiment': exp['name'],
                        'params': linear_params,
                        'r2': r2_linear,
                        'air_y': air_y_valid,
                        'water_y': water_y_valid,
                        'equation': f"Y_water = {linear_params[0]:.4f} * Y_air + {linear_params[1]:.4f}"
                    })
                    
                    # Plot 1: Individual linear relationships
                    ax1.scatter(air_y_valid, water_y_valid, alpha=0.6, s=20, color=colors[i], 
                              label=f"{exp['name']} (R²={r2_linear:.3f})")
                    ax1.plot(air_y_valid, y_pred, color=colors[i], linewidth=2, alpha=0.8)
            except:
                continue
        
        ax1.set_title(f'Linear Relationships Between {title} Curves')
        ax1.set_xlabel('Air Pressure Drop (from polynomial)')
        ax1.set_ylabel('Water Pressure Drop (from polynomial)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: R² distribution of linear relationships
        if linear_relationships:
            r2_values = [rel['r2'] for rel in linear_relationships]
            exp_names = [rel['experiment'] for rel in linear_relationships]
            
            bars = ax2.bar(range(len(r2_values)), r2_values, color=colors[:len(r2_values)], alpha=0.7)
            ax2.set_title(f'Linear Relationship Quality ({title})')
            ax2.set_xlabel('Experiment')
            ax2.set_ylabel('R² of Linear Relationship')
            ax2.set_xticks(range(len(exp_names)))
            ax2.set_xticklabels(exp_names, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Add R² values on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{r2_values[i]:.3f}', ha='center', va='bottom')
        
        # Plot 3: Slope comparison
        if linear_relationships:
            slopes = [rel['params'][0] for rel in linear_relationships]
            intercepts = [rel['params'][1] for rel in linear_relationships]
            
            ax3.scatter(slopes, intercepts, s=100, alpha=0.7, c=range(len(slopes)), cmap='tab10')
            
            for i, rel in enumerate(linear_relationships):
                ax3.annotate(rel['experiment'], (slopes[i], intercepts[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax3.set_title('Linear Relationship Parameters')
            ax3.set_xlabel('Slope (a)')
            ax3.set_ylabel('Intercept (b)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined linear relationships
        if linear_relationships:
            # Combine all data points
            all_air_y = []
            all_water_y = []
            
            for rel in linear_relationships:
                all_air_y.extend(rel['air_y'])
                all_water_y.extend(rel['water_y'])
            
            if all_air_y and all_water_y:
                # Fit overall linear relationship
                overall_params = np.polyfit(all_air_y, all_water_y, 1)
                y_pred_overall = np.polyval(overall_params, all_air_y)
                r2_overall = r2_score(all_water_y, y_pred_overall)
                
                ax4.scatter(all_air_y, all_water_y, alpha=0.5, s=10, color='gray', label='All Data Points')
                
                air_range = np.linspace(min(all_air_y), max(all_air_y), 100)
                water_pred = np.polyval(overall_params, air_range)
                ax4.plot(air_range, water_pred, 'r-', linewidth=3, 
                        label=f'Combined Linear Fit (R²={r2_overall:.3f})')
                
                ax4.set_title(f'Combined Linear Relationship ({title})')
                ax4.set_xlabel('Air Pressure Drop')
                ax4.set_ylabel('Water Pressure Drop')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Add equation text
                equation_text = f"Y_water = {overall_params[0]:.4f} * Y_air + {overall_params[1]:.4f}"
                ax4.text(0.05, 0.95, equation_text, transform=ax4.transAxes, 
                        bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        fig.tight_layout()
        canvas.draw()
        
        # Store linear relationships for later use
        setattr(self, f'{poly_type}_linear_relationships', linear_relationships)
        
    def export_polynomial_results(self, poly_experiments, title, poly_type):
        """Export polynomial relationships analysis"""
        file_path = filedialog.asksaveasfilename(
            title=f"Export {title} Relationships Analysis",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    for exp in poly_experiments:
                        water_poly = exp['water_fit_results'][poly_type]
                        air_poly = exp['air_fit_results'][poly_type]
                        
                        row = {
                            'Experiment_Name': exp['name'],
                            f'Water_{title}_R2': water_poly['r2'],
                            f'Air_{title}_R2': air_poly['r2'],
                            'Water_Coefficients': str(water_poly['params']),
                            'Air_Coefficients': str(air_poly['params']),
                            'Quality': 'GOOD' if min(water_poly['r2'], air_poly['r2']) > 0.8 else 
                                      'ACCEPTABLE' if min(water_poly['r2'], air_poly['r2']) > 0.7 else 'POOR'
                        }
                        
                        summary_data.append(row)
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name=f'{poly_type}_Analysis', index=False)
                    
                    # Linear relationships between polynomials
                    if hasattr(self, f'{poly_type}_linear_relationships'):
                        linear_rels = getattr(self, f'{poly_type}_linear_relationships')
                        if linear_rels:
                            linear_data = []
                            for rel in linear_rels:
                                linear_data.append({
                                    'Experiment': rel['experiment'],
                                    'Linear_R2': rel['r2'],
                                    'Slope': rel['params'][0],
                                    'Intercept': rel['params'][1],
                                    'Equation': rel['equation']
                                })
                            
                            linear_df = pd.DataFrame(linear_data)
                            linear_df.to_excel(writer, sheet_name=f'{poly_type}_Linear_Relationships', index=False)
                
                messagebox.showinfo("Success", f"{title} relationships analysis exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting {title} analysis:\n{str(e)}")
        
    def calculate_average_polynomial2_relationship(self):
        """Calculate average polynomial degree 2 relationship"""
        self.calculate_average_polynomial_relationship_generic("poly2", "Polynomial Degree 2")
        
    def calculate_average_polynomial3_relationship(self):
        """Calculate average polynomial degree 3 relationship"""
        self.calculate_average_polynomial_relationship_generic("poly3", "Polynomial Degree 3")
        
    def calculate_average_polynomial_relationship_generic(self, poly_type, title):
        """Generic function to calculate average polynomial relationship"""
        # Get all experiments with polynomial relationships and their linear relationships
        poly_experiments = []
        for exp in self.experiments:
            if (exp.get('water_fit_results') and poly_type in exp['water_fit_results'] and
                exp.get('air_fit_results') and poly_type in exp['air_fit_results']):
                poly_experiments.append(exp)
        
        if len(poly_experiments) < 2:
            messagebox.showwarning("Insufficient Data", 
                                  f"Need at least 2 experiments with {title} relationships to calculate average.")
            return
        
        # Calculate linear relationships between polynomial curves
        linear_relationships = []
        
        for exp in poly_experiments:
            water_poly = exp['water_fit_results'][poly_type]
            air_poly = exp['air_fit_results'][poly_type]
            
            # Generate comparison points
            x_min = max(np.min(water_poly['x_data']), np.min(air_poly['x_data']))
            x_max = min(np.max(water_poly['x_data']), np.max(air_poly['x_data']))
            
            x_comparison = np.linspace(x_min, x_max, 100)
            
            water_y = water_poly['function'](x_comparison)
            air_y = air_poly['function'](x_comparison)
            
            # Fit linear relationship: water_y = a * air_y + b
            try:
                # Remove any invalid values
                valid_mask = np.isfinite(water_y) & np.isfinite(air_y)
                if np.sum(valid_mask) > 10:  # Need enough points
                    water_y_valid = water_y[valid_mask]
                    air_y_valid = air_y[valid_mask]
                    
                    # Fit linear relationship
                    linear_params = np.polyfit(air_y_valid, water_y_valid, 1)
                    y_pred = np.polyval(linear_params, air_y_valid)
                    r2_linear = r2_score(water_y_valid, y_pred)
                    
                    linear_relationships.append({
                        'experiment': exp['name'],
                        'params': linear_params,
                        'r2': r2_linear,
                        'equation': f"Y_water = {linear_params[0]:.4f} * Y_air + {linear_params[1]:.4f}"
                    })
            except:
                continue
        
        if len(linear_relationships) < 2:
            messagebox.showwarning("Insufficient Linear Relationships", 
                                  f"Could not find enough valid linear relationships between {title} curves.")
            return
        
        # Calculate weighted average (weighted by R²)
        relationship_params = [rel['params'] for rel in linear_relationships]
        relationship_r2_values = [rel['r2'] for rel in linear_relationships]
        
        weights = np.array(relationship_r2_values)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Average parameters
        avg_params = np.average(relationship_params, axis=0, weights=weights)
        avg_r2 = np.mean(relationship_r2_values)
        
        # Create average relationship function
        def avg_relationship_func(y_air):
            """Average relationship function to estimate water pressure drop from air pressure drop"""
            return avg_params[0] * y_air + avg_params[1]
        
        # Store the average relationship
        avg_relationship = {
            'poly_type': poly_type,
            'title': title,
            'params': avg_params,
            'linear_params': avg_params,  # Store linear parameters for water estimation columns
            'r2_avg': avg_r2,
            'function': avg_relationship_func,
            'equation': f"Y_water = {avg_params[0]:.4f} * Y_air + {avg_params[1]:.4f}",
            'n_experiments': len(linear_relationships),
            'weights': weights,
            'individual_r2': relationship_r2_values,
            'individual_relationships': linear_relationships
        }
        
        setattr(self, f'average_{poly_type}_relationship', avg_relationship)
        
        # Show results
        self.show_average_polynomial_relationship_results(avg_relationship, poly_experiments)
    
    def show_average_polynomial_relationship_results(self, avg_relationship, poly_experiments):
        """Show the results of the average polynomial relationship calculation"""
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Average {avg_relationship['title']} Linear Relationship for Water Pressure Drop Estimation")
        results_window.geometry("1200x800")
        results_window.transient(self.root)
        
        # Create notebook
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Results Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Average Relationship Results")
        
        # Tab 2: Validation Plot
        validation_frame = ttk.Frame(notebook)
        notebook.add(validation_frame, text="Validation & Prediction")
        
        # Tab 3: Usage Guide
        guide_frame = ttk.Frame(notebook)
        notebook.add(guide_frame, text="Usage Guide")
        
        # Create results summary
        self.create_average_polynomial_relationship_summary(summary_frame, avg_relationship, poly_experiments)
        self.create_polynomial_validation_plot(validation_frame, avg_relationship, poly_experiments)
        self.create_polynomial_usage_guide(guide_frame, avg_relationship)
        
        # Control buttons
        control_frame = ttk.Frame(results_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_frame, text=f"Export Average {avg_relationship['title']} Relationship", 
                  command=lambda: self.export_average_polynomial_relationship(avg_relationship)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Test Prediction", 
                  command=lambda: self.test_polynomial_prediction(avg_relationship)).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", 
                  command=results_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def create_average_polynomial_relationship_summary(self, parent, avg_relationship, poly_experiments):
        """Create summary of the average polynomial relationship"""
        # Main frame with text widget
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=('Courier', 11))
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create summary content
        content = f"AVERAGE LINEAR RELATIONSHIP FROM {avg_relationship['title'].upper()} CURVES\n"
        content += "=" * 80 + "\n\n"
        
        content += f"📊 MAIN EQUATION (for estimation from {avg_relationship['title']} curves):\n"
        content += f"   {avg_relationship['equation']}\n\n"
        
        content += f"📈 STATISTICAL SUMMARY:\n"
        content += f"   • Number of {avg_relationship['title']} experiments used: {avg_relationship['n_experiments']}\n"
        content += f"   • Average R² of linear relationships: {avg_relationship['r2_avg']:.4f}\n"
        content += f"   • Method: Linear relationship between {avg_relationship['title']} curves\n\n"
        
        content += f"📋 INDIVIDUAL EXPERIMENT CONTRIBUTIONS:\n"
        content += "   Exp.  Weight   R²     Linear Equation\n"
        content += "   " + "-" * 60 + "\n"
        
        for i, rel in enumerate(avg_relationship['individual_relationships']):
            weight = avg_relationship['weights'][i]
            r2 = rel['r2']
            params = rel['params']
            eq_short = f"Y = {params[0]:.3f}*X + {params[1]:.3f}"
            content += f"   {i+1:2d}.   {weight:.3f}   {r2:.3f}   {eq_short}\n"
        
        content += f"\n🎯 ACCURACY ASSESSMENT:\n"
        r2_values = avg_relationship['individual_r2']
        content += f"   • Best individual R²: {max(r2_values):.4f}\n"
        content += f"   • Worst individual R²: {min(r2_values):.4f}\n"
        content += f"   • Standard deviation of R²: {np.std(r2_values):.4f}\n"
        
        # Quality rating
        if avg_relationship['r2_avg'] > 0.9:
            quality = "EXCELLENT"
        elif avg_relationship['r2_avg'] > 0.8:
            quality = "GOOD"
        elif avg_relationship['r2_avg'] > 0.7:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR"
        
        content += f"   • Overall quality rating: {quality}\n"
        
        content += f"\n🔧 USAGE INSTRUCTIONS:\n"
        content += f"   1. Fit {avg_relationship['title']} to your air pressure drop data\n"
        content += f"   2. Calculate air pressure drop values using the {avg_relationship['title']} fit\n"
        content += f"   3. Apply the equation: Y_water = {avg_relationship['params'][0]:.4f} * Y_air + {avg_relationship['params'][1]:.4f}\n"
        content += f"   4. The result gives you the estimated water pressure drop\n"
        content += f"   5. Expected accuracy: ±{(1-avg_relationship['r2_avg'])*100:.1f}% based on R² = {avg_relationship['r2_avg']:.3f}\n"
        
        content += f"\n⚠️  IMPORTANT NOTES:\n"
        content += f"   • This relationship is based on {avg_relationship['title']} curve fitting\n"
        content += f"   • Both air and water must follow {avg_relationship['title']} behavior\n"
        content += f"   • Valid for similar operating conditions and system geometry\n"
        content += f"   • More complex than linear analysis but may be more accurate for non-linear systems\n"
        
        content += f"\n📊 COMPARISON WITH LINEAR ANALYSIS:\n"
        content += f"   • {avg_relationship['title']} approach captures non-linear behavior\n"
        content += f"   • Linear relationship found between {avg_relationship['title']} curves\n"
        content += f"   • May provide better accuracy for complex flow patterns\n"
        content += f"   • Requires good {avg_relationship['title']} fits (R² > 0.8 recommended)\n"
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def create_polynomial_validation_plot(self, parent, avg_relationship, poly_experiments):
        """Create validation plot for polynomial relationship"""
        # Create matplotlib figure
        fig = Figure(figsize=(14, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(avg_relationship['individual_relationships'])))
        
        # Collect validation data
        all_air_y = []
        all_water_y = []
        all_predicted_y = []
        
        # Plot 1: Individual relationships vs average
        for i, rel in enumerate(avg_relationship['individual_relationships']):
            air_y = rel['air_y']
            water_y = rel['water_y']
            
            all_air_y.extend(air_y)
            all_water_y.extend(water_y)
            
            # Plot individual relationship points
            ax1.scatter(air_y, water_y, alpha=0.6, s=20, color=colors[i], 
                      label=f"{rel['experiment']} (R²={rel['r2']:.3f})")
            
            # Individual fit line
            air_range = np.linspace(np.min(air_y), np.max(air_y), 50)
            water_individual = rel['params'][0] * air_range + rel['params'][1]
            ax1.plot(air_range, water_individual, color=colors[i], linewidth=1, alpha=0.8)
            
            # Predictions using average relationship
            predicted_y = avg_relationship['function'](air_y)
            all_predicted_y.extend(predicted_y)
        
        # Plot average relationship line
        if all_air_y:
            air_overall_range = np.linspace(min(all_air_y), max(all_air_y), 100)
            water_avg_pred = avg_relationship['function'](air_overall_range)
            ax1.plot(air_overall_range, water_avg_pred, 'k-', linewidth=3, alpha=0.9, 
                    label=f'Average Relationship (R²_avg={avg_relationship["r2_avg"]:.3f})')
        
        ax1.set_title(f'Individual vs Average {avg_relationship["title"]} Relationship')
        ax1.set_xlabel(f'Air Pressure Drop (from {avg_relationship["title"]})')
        ax1.set_ylabel(f'Water Pressure Drop (from {avg_relationship["title"]})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction accuracy
        if all_water_y and all_predicted_y:
            ax2.scatter(all_water_y, all_predicted_y, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(min(all_water_y), min(all_predicted_y))
            max_val = max(max(all_water_y), max(all_predicted_y))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
            
            # Calculate prediction R²
            pred_r2 = r2_score(all_water_y, all_predicted_y)
            
            ax2.set_title(f'Prediction Accuracy (R² = {pred_r2:.4f})')
            ax2.set_xlabel('Actual Water Pressure Drop')
            ax2.set_ylabel('Predicted Water Pressure Drop')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            residuals = np.array(all_predicted_y) - np.array(all_water_y)
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            
            ax2.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
                    transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
        
        # Plot 3: R² comparison
        exp_names = [rel['experiment'] for rel in avg_relationship['individual_relationships']]
        r2_values = [rel['r2'] for rel in avg_relationship['individual_relationships']]
        
        bars = ax3.bar(range(len(r2_values)), r2_values, color=colors, alpha=0.7)
        ax3.axhline(y=avg_relationship['r2_avg'], color='red', linestyle='--', linewidth=2, label=f'Average R² = {avg_relationship["r2_avg"]:.3f}')
        
        ax3.set_title(f'{avg_relationship["title"]} Linear Relationship Quality')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('R² of Linear Relationship')
        ax3.set_xticks(range(len(exp_names)))
        ax3.set_xticklabels(exp_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Error distribution
        if all_water_y and all_predicted_y:
            residuals = np.array(all_predicted_y) - np.array(all_water_y)
            relative_errors = 100 * residuals / (np.array(all_water_y) + 1e-10)
            
            ax4.hist(relative_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(x=0, color='r', linestyle='--', alpha=0.8)
            ax4.axvline(x=np.mean(relative_errors), color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(relative_errors):.2f}%')
            
            ax4.set_title('Prediction Error Distribution')
            ax4.set_xlabel('Relative Error (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas.draw()
    
    def create_polynomial_usage_guide(self, parent, avg_relationship):
        """Create usage guide for the average polynomial relationship"""
        # Create text widget with instructions
        text_widget = tk.Text(parent, wrap=tk.WORD, font=('Arial', 11))
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        guide_content = f"""HOW TO USE THE AVERAGE {avg_relationship['title'].upper()} RELATIONSHIP FOR WATER PRESSURE DROP ESTIMATION

🎯 PURPOSE:
This tool provides you with a calibrated equation to estimate water pressure drop based on {avg_relationship['title']} fitted air pressure drop curves. This approach:
• Captures non-linear behavior through {avg_relationship['title']} fitting
• Finds linear relationships between the polynomial curves
• Provides potentially better accuracy for complex flow patterns

📐 THE EQUATION:
Your calibrated equation is: {avg_relationship['equation']}

🔧 STEP-BY-STEP USAGE:

1. MEASURE AIR PRESSURE DROP DATA:
   • Set up your system with air as the working fluid
   • Measure pressure drop at multiple flow rates (need sufficient data for {avg_relationship['title']} fitting)
   • Ensure you have enough points for good polynomial fitting

2. FIT {avg_relationship['title'].upper()} TO AIR DATA:
   • Use the curve fitting feature to fit {avg_relationship['title']} to your air data
   • Ensure good fit quality (R² > 0.8 recommended)
   • Note the {avg_relationship['title']} parameters

3. CALCULATE AIR PRESSURE DROP FROM POLYNOMIAL:
   • Use your {avg_relationship['title']} fit to calculate air pressure drop values
   • This gives you Y_air values from the polynomial relationship

4. APPLY THE LINEAR EQUATION:
   • Substitute your Y_air values into: Y_water = {avg_relationship['params'][0]:.4f} * Y_air + {avg_relationship['params'][1]:.4f}
   • The result is your estimated water pressure drop

5. EXAMPLE CALCULATION:
   If your {avg_relationship['title']} fit gives an air pressure drop of 100 Pa:
   Y_water = {avg_relationship['params'][0]:.4f} * 100 + {avg_relationship['params'][1]:.4f}
   Y_water = {avg_relationship['params'][0] * 100 + avg_relationship['params'][1]:.2f} Pa

⚠️ IMPORTANT CONSIDERATIONS:

• POLYNOMIAL FIT QUALITY: Ensure your air data follows {avg_relationship['title']} behavior (R² > 0.8)
• SIMILAR SYSTEMS: Most accurate for geometrically similar systems
• FLOW PATTERNS: Best for systems with similar flow characteristics
• DATA RANGE: Stay within the range used for calibration

📊 ACCURACY EXPECTATIONS:
• Expected accuracy: ±{(1-avg_relationship['r2_avg'])*100:.1f}% based on R² = {avg_relationship['r2_avg']:.3f}
• Based on {avg_relationship['n_experiments']} calibration experiments
• Quality rating: {('EXCELLENT' if avg_relationship['r2_avg'] > 0.9 else 'GOOD' if avg_relationship['r2_avg'] > 0.8 else 'ACCEPTABLE' if avg_relationship['r2_avg'] > 0.7 else 'POOR')}

🔬 ADVANTAGES OVER LINEAR ANALYSIS:
• Captures non-linear pressure drop behavior
• Better for complex flow patterns (turbulent, transitional)
• Accounts for velocity-dependent effects
• More accurate for systems with significant non-linearity

🎯 WHEN TO USE THIS METHOD:
• When linear analysis gives poor fits (R² < 0.8)
• For systems with clear non-linear behavior
• When you need higher accuracy
• For complex geometries or flow patterns

💡 TIPS FOR BEST RESULTS:
• Ensure good {avg_relationship['title']} fits for both calibration and application
• Use similar flow rate ranges between air and water tests
• Validate with actual water measurements when possible
• Consider Reynolds number similarity
• Document operating conditions for reproducibility

📁 EXPORT OPTIONS:
• Use "Export Average {avg_relationship['title']} Relationship" to save the equation
• Include polynomial fitting parameters in documentation
• Save validation results for quality assurance
"""
        
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED)
    
    def export_average_polynomial_relationship(self, avg_relationship):
        """Export the average polynomial relationship equation and parameters"""
        file_path = filedialog.asksaveasfilename(
            title=f"Export Average {avg_relationship['title']} Relationship",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    # Export to Excel
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Main equation sheet
                        equation_data = {
                            'Parameter': [
                                'Analysis Type',
                                'Equation',
                                'Slope (a)',
                                'Intercept (b)', 
                                'Average R²',
                                'Number of Experiments',
                                'Quality Rating'
                            ],
                            'Value': [
                                f'Average Linear Relationship from {avg_relationship["title"]} Curves',
                                avg_relationship['equation'],
                                avg_relationship['params'][0],
                                avg_relationship['params'][1],
                                avg_relationship['r2_avg'],
                                avg_relationship['n_experiments'],
                                'EXCELLENT' if avg_relationship['r2_avg'] > 0.9 else 
                                'GOOD' if avg_relationship['r2_avg'] > 0.8 else 
                                'ACCEPTABLE' if avg_relationship['r2_avg'] > 0.7 else 'POOR'
                            ]
                        }
                        
                        equation_df = pd.DataFrame(equation_data)
                        equation_df.to_excel(writer, sheet_name=f'Average_{avg_relationship["poly_type"]}_Relationship', index=False)
                        
                        # Individual contributions
                        contrib_data = []
                        for i, rel in enumerate(avg_relationship['individual_relationships']):
                            contrib_data.append({
                                'Experiment': rel['experiment'],
                                'Weight': avg_relationship['weights'][i],
                                'Individual_R2': rel['r2'],
                                'Individual_Equation': rel['equation']
                            })
                        
                        contrib_df = pd.DataFrame(contrib_data)
                        contrib_df.to_excel(writer, sheet_name='Individual_Contributions', index=False)
                        
                        # Usage instructions
                        instructions = pd.DataFrame({
                            'Step': [
                                '1. Fit polynomial to air data',
                                '2. Calculate air pressure drop',
                                '3. Apply linear equation',
                                '4. Get water estimate',
                                '5. Validate result'
                            ],
                            'Description': [
                                f'Fit {avg_relationship["title"]} to your air pressure drop data',
                                f'Use {avg_relationship["title"]} fit to calculate Y_air values',
                                f'Y_water = {avg_relationship["params"][0]:.4f} * Y_air + {avg_relationship["params"][1]:.4f}',
                                'Calculate Y_water using the equation above',
                                f'Expected accuracy: ±{(1-avg_relationship["r2_avg"])*100:.1f}%'
                            ]
                        })
                        
                        instructions.to_excel(writer, sheet_name='Usage_Instructions', index=False)
                    
                else:
                    # Export to text file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"AVERAGE {avg_relationship['title'].upper()} RELATIONSHIP FOR WATER PRESSURE DROP ESTIMATION\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"EQUATION: {avg_relationship['equation']}\n\n")
                        f.write(f"Parameters:\n")
                        f.write(f"  Slope (a): {avg_relationship['params'][0]:.6f}\n")
                        f.write(f"  Intercept (b): {avg_relationship['params'][1]:.6f}\n")
                        f.write(f"  Average R²: {avg_relationship['r2_avg']:.4f}\n")
                        f.write(f"  Based on {avg_relationship['n_experiments']} experiments\n\n")
                        f.write("USAGE:\n")
                        f.write(f"1. Fit {avg_relationship['title']} to air pressure drop data\n")
                        f.write(f"2. Calculate air pressure drop from polynomial: Y_air\n")
                        f.write(f"3. Calculate: Y_water = {avg_relationship['params'][0]:.4f} * Y_air + {avg_relationship['params'][1]:.4f}\n")
                        f.write(f"4. Expected accuracy: ±{(1-avg_relationship['r2_avg'])*100:.1f}%\n")
                
                messagebox.showinfo("Success", f"Average {avg_relationship['title']} relationship exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting average {avg_relationship['title']} relationship:\n{str(e)}")
    
    def test_polynomial_prediction(self, avg_relationship):
        """Test the polynomial prediction with user-provided air pressure drop value"""
        # Ask user for air pressure drop value
        air_value = tk.simpledialog.askfloat(
            "Test Polynomial Prediction",
            f"Enter air pressure drop value (from {avg_relationship['title']} fit) to predict water pressure drop:",
            minvalue=0.0
        )
        
        if air_value is not None:
            # Calculate prediction
            predicted_water = avg_relationship['function'](air_value)
            
            # Show results
            result_msg = f"INPUT: Air pressure drop (from {avg_relationship['title']}) = {air_value}\n\n"
            result_msg += f"PREDICTION: Water pressure drop = {predicted_water:.4f}\n\n"
            result_msg += f"EQUATION USED: {avg_relationship['equation']}\n\n"
            result_msg += f"CALCULATION:\n"
            result_msg += f"Y_water = {avg_relationship['params'][0]:.4f} × {air_value} + {avg_relationship['params'][1]:.4f}\n"
            result_msg += f"Y_water = {predicted_water:.4f}\n\n"
            result_msg += f"METHOD: Linear relationship between {avg_relationship['title']} curves\n"
            result_msg += f"ACCURACY: Expected ±{(1-avg_relationship['r2_avg'])*100:.1f}% based on R² = {avg_relationship['r2_avg']:.3f}"
            
            messagebox.showinfo("Polynomial Prediction Result", result_msg)
        
    # Export methods
    def export_polynomial_results(self, poly_type):
        """Export polynomial analysis results to Excel"""
        if not hasattr(self, 'experiments') or not self.experiments:
            messagebox.showwarning("No Data", "No experiments loaded for export.")
            return
        
        # Find experiments with polynomial fits
        poly_experiments = []
        for exp in self.experiments:
            water_fits = exp.get('water_fit_results', {})
            air_fits = exp.get('air_fit_results', {})
            
            if f'poly_{poly_type}' in water_fits and f'poly_{poly_type}' in air_fits:
                poly_experiments.append(exp)
        
        if not poly_experiments:
            messagebox.showwarning("No Data", f"No experiments with polynomial degree {poly_type} fits found.")
            return
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            title=f"Export Polynomial Degree {poly_type} Analysis Results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                
                for exp in poly_experiments:
                    water_fit = exp['water_fit_results'][f'poly_{poly_type}']
                    air_fit = exp['air_fit_results'][f'poly_{poly_type}']
                    
                    # Calculate linear relationship if data is available
                    linear_r2 = 'N/A'
                    linear_equation = 'N/A'
                    
                    if exp.get('water_data') is not None and exp.get('air_data') is not None:
                        water_cols = exp.get('water_columns', ('', ''))
                        air_cols = exp.get('air_columns', ('', ''))
                        
                        if water_cols[0] and water_cols[1] and air_cols[0] and air_cols[1]:
                            try:
                                water_x = exp['water_data'][water_cols[0]].values
                                air_x = exp['air_data'][air_cols[0]].values
                                
                                air_x_range = np.linspace(np.min(air_x), np.max(air_x), 50)
                                water_x_range = np.linspace(np.min(water_x), np.max(water_x), 50)
                                
                                air_y = np.polyval(air_fit['params'], air_x_range)
                                water_y = np.polyval(water_fit['params'], water_x_range)
                                
                                # Find linear relationship
                                poly_coeffs = np.polyfit(air_y, water_y, 1)
                                linear_r2 = r2_score(water_y, np.polyval(poly_coeffs, air_y))
                                linear_equation = f"Y = {poly_coeffs[0]:.4f}*X + {poly_coeffs[1]:.4f}"
                            except:
                                pass
                    
                    summary_data.append({
                        'Experiment': exp['name'],
                        'Water_R2': water_fit['r2'],
                        'Air_R2': air_fit['r2'],
                        'Linear_Slope': poly_coeffs[0] if 'poly_coeffs' in locals() else 'N/A',
                        'Linear_Intercept': poly_coeffs[1] if 'poly_coeffs' in locals() else 'N/A',
                        'Linear_R2': linear_r2,
                        'Linear_Equation': linear_equation
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name=f'Poly{poly_type}_Summary', index=False)
                
                # Individual experiment details
                for exp in poly_experiments:
                    water_fit = exp['water_fit_results'][f'poly_{poly_type}']
                    air_fit = exp['air_fit_results'][f'poly_{poly_type}']
                    
                    # Polynomial parameters
                    params_data = {
                        'Parameter': [
                            'Water_Polynomial_R2',
                            'Air_Polynomial_R2'
                        ] + [f'Water_Poly_Coeff_{i}' for i in range(len(water_fit['params']))] + 
                        [f'Air_Poly_Coeff_{i}' for i in range(len(air_fit['params']))],
                        
                        'Value': [
                            water_fit['r2'],
                            air_fit['r2']
                        ] + list(water_fit['params']) + list(air_fit['params'])
                    }
                    
                    params_df = pd.DataFrame(params_data)
                    
                    # Create safe sheet name
                    safe_name = exp['name'].replace('/', '_').replace('\\', '_')[:31]
                    params_df.to_excel(writer, sheet_name=safe_name, index=False)
                
                # Analysis metadata
                metadata = pd.DataFrame({
                    'Analysis_Info': [
                        'Polynomial_Degree',
                        'Number_of_Experiments',
                        'Analysis_Date',
                        'Analysis_Type'
                    ],
                    'Value': [
                        poly_type,
                        len(poly_experiments),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        f'Polynomial Degree {poly_type} Analysis with Linear Relationships'
                    ]
                })
                
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            messagebox.showinfo("Success", f"Polynomial degree {poly_type} analysis exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting polynomial analysis:\n{str(e)}")
    
    def get_polynomial_summary_for_report(self, poly_type):
        """Get polynomial analysis summary for inclusion in reports"""
        if not hasattr(self, 'experiments') or not self.experiments:
            return None
        
        # Find experiments with polynomial fits
        poly_experiments = []
        for exp in self.experiments:
            water_fits = exp.get('water_fit_results', {})
            air_fits = exp.get('air_fit_results', {})
            
            if f'poly_{poly_type}' in water_fits and f'poly_{poly_type}' in air_fits:
                poly_experiments.append(exp)
        
        if not poly_experiments:
            return None
        
        # Calculate linear relationships
        relationships = []
        for exp in poly_experiments:
            water_fit = exp['water_fit_results'][f'poly_{poly_type}']
            air_fit = exp['air_fit_results'][f'poly_{poly_type}']
            
            # Generate points from polynomial fits
            if exp.get('water_data') is not None and exp.get('air_data') is not None:
                water_cols = exp.get('water_columns', ('', ''))
                air_cols = exp.get('air_columns', ('', ''))
                
                if water_cols[0] and water_cols[1] and air_cols[0] and air_cols[1]:
                    water_x = exp['water_data'][water_cols[0]].values
                    air_x = exp['air_data'][air_cols[0]].values
                    
                    air_x_range = np.linspace(np.min(air_x), np.max(air_x), 50)
                    water_x_range = np.linspace(np.min(water_x), np.max(water_x), 50)
                    
                    air_y = np.polyval(air_fit['params'], air_x_range)
                    water_y = np.polyval(water_fit['params'], water_x_range)
                    
                    # Find linear relationship
                    poly_coeffs = np.polyfit(air_y, water_y, 1)
                    r2 = r2_score(water_y, np.polyval(poly_coeffs, air_y))
                    
                    relationships.append({
                        'experiment': exp['name'],
                        'water_r2': water_fit['r2'],
                        'air_r2': air_fit['r2'],
                        'linear_slope': poly_coeffs[0],
                        'linear_intercept': poly_coeffs[1],
                        'linear_r2': r2,
                        'equation': f"Y = {poly_coeffs[0]:.4f}*X + {poly_coeffs[1]:.4f}"
                    })
        
        if not relationships:
            return None
        
        # Calculate average relationship
        avg_slope = np.mean([rel['linear_slope'] for rel in relationships])
        avg_intercept = np.mean([rel['linear_intercept'] for rel in relationships])
        avg_r2 = np.mean([rel['linear_r2'] for rel in relationships])
        
        return {
            'poly_type': poly_type,
            'title': f"Polynomial Degree {poly_type}",
            'n_experiments': len(poly_experiments),
            'relationships': relationships,
            'average_equation': f"Y = {avg_slope:.4f}*X + {avg_intercept:.4f}",
            'average_slope': avg_slope,
            'average_intercept': avg_intercept,
            'average_r2': avg_r2,
            'min_r2': min([rel['linear_r2'] for rel in relationships]),
            'max_r2': max([rel['linear_r2'] for rel in relationships]),
            'std_r2': np.std([rel['linear_r2'] for rel in relationships])
        }
    
    def add_polynomial_analysis_to_summary(self, content):
        """Add polynomial analysis section to summary report"""
        # Add section for polynomial degree 2
        poly2_summary = self.get_polynomial_summary_for_report(2)
        if poly2_summary:
            content += "\n" + "="*80 + "\n"
            content += "POLYNOMIAL DEGREE 2 ANALYSIS - LINEAR RELATIONSHIPS\n"
            content += "="*80 + "\n\n"
            
            content += f"📊 OVERVIEW:\n"
            content += f"   • Analysis of linear relationships between polynomial degree 2 curves\n"
            content += f"   • Number of experiments with valid poly-2 fits: {poly2_summary['n_experiments']}\n"
            content += f"   • Average linear relationship: {poly2_summary['average_equation']}\n"
            content += f"   • Average R² of linear relationships: {poly2_summary['average_r2']:.4f}\n\n"
            
            content += f"📈 STATISTICAL SUMMARY:\n"
            content += f"   • Best linear R²: {poly2_summary['max_r2']:.4f}\n"
            content += f"   • Worst linear R²: {poly2_summary['min_r2']:.4f}\n"
            content += f"   • Standard deviation: {poly2_summary['std_r2']:.4f}\n"
            content += f"   • Quality: {'EXCELLENT' if poly2_summary['average_r2'] > 0.9 else 'GOOD' if poly2_summary['average_r2'] > 0.8 else 'ACCEPTABLE' if poly2_summary['average_r2'] > 0.7 else 'POOR'}\n\n"
            
            content += f"📋 INDIVIDUAL EXPERIMENT RESULTS:\n"
            content += f"   Experiment                    Water R²  Air R²   Linear R²  Linear Equation\n"
            content += f"   " + "-"*80 + "\n"
            
            for rel in poly2_summary['relationships']:
                exp_name = rel['experiment'][:25].ljust(25)
                content += f"   {exp_name}  {rel['water_r2']:.3f}     {rel['air_r2']:.3f}    {rel['linear_r2']:.3f}      {rel['equation']}\n"
        
        # Add section for polynomial degree 3
        poly3_summary = self.get_polynomial_summary_for_report(3)
        if poly3_summary:
            content += "\n" + "="*80 + "\n"
            content += "POLYNOMIAL DEGREE 3 ANALYSIS - LINEAR RELATIONSHIPS\n"
            content += "="*80 + "\n\n"
            
            content += f"📊 OVERVIEW:\n"
            content += f"   • Analysis of linear relationships between polynomial degree 3 curves\n"
            content += f"   • Number of experiments with valid poly-3 fits: {poly3_summary['n_experiments']}\n"
            content += f"   • Average linear relationship: {poly3_summary['average_equation']}\n"
            content += f"   • Average R² of linear relationships: {poly3_summary['average_r2']:.4f}\n\n"
            
            content += f"📈 STATISTICAL SUMMARY:\n"
            content += f"   • Best linear R²: {poly3_summary['max_r2']:.4f}\n"
            content += f"   • Worst linear R²: {poly3_summary['min_r2']:.4f}\n"
            content += f"   • Standard deviation: {poly3_summary['std_r2']:.4f}\n"
            content += f"   • Quality: {'EXCELLENT' if poly3_summary['average_r2'] > 0.9 else 'GOOD' if poly3_summary['average_r2'] > 0.8 else 'ACCEPTABLE' if poly3_summary['average_r2'] > 0.7 else 'POOR'}\n\n"
            
            content += f"📋 INDIVIDUAL EXPERIMENT RESULTS:\n"
            content += f"   Experiment                    Water R²  Air R²   Linear R²  Linear Equation\n"
            content += f"   " + "-"*80 + "\n"
            
            for rel in poly3_summary['relationships']:
                exp_name = rel['experiment'][:25].ljust(25)
                content += f"   {exp_name}  {rel['water_r2']:.3f}     {rel['air_r2']:.3f}    {rel['linear_r2']:.3f}      {rel['equation']}\n"
        
        return content
    def export_to_excel(self):
        """Export results to Excel"""
        if not (self.water_fit_results or self.air_fit_results):
            messagebox.showwarning("No Data", "No results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save results to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Water results
                    if self.water_fit_results:
                        water_summary = self.create_results_dataframe(self.water_fit_results, "Agua")
                        water_summary.to_excel(writer, sheet_name='Water_Results', index=False)
                        
                    # Air results
                    if self.air_fit_results:
                        air_summary = self.create_results_dataframe(self.air_fit_results, "Aire")
                        air_summary.to_excel(writer, sheet_name='Air_Results', index=False)
                        
                    # Relationship results
                    if self.relationship_results:
                        rel_summary = self.create_relationship_dataframe()
                        rel_summary.to_excel(writer, sheet_name='Relationships', index=False)
                        
                messagebox.showinfo("Success", f"Results exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting:\n{str(e)}")
                
    def create_results_dataframe(self, fit_results, data_type):
        """Create DataFrame with fitting results"""
        results_list = []
        
        for model_name, results in fit_results.items():
            results_list.append({
                'Modelo': model_name,
                'R²': results['r2'],
                'Parámetros': str(results['params'])
            })
            
        return pd.DataFrame(results_list)
        
    def create_relationship_dataframe(self):
        """Create DataFrame with relationship results"""
        if not self.relationship_results:
            return pd.DataFrame()
            
        rel = self.relationship_results
        data = []
        
        data.append({
            'Aspecto': 'Modelo Agua',
            'Valor': rel['water_model']
        })
        
        data.append({
            'Aspecto': 'Modelo Aire', 
            'Valor': rel['air_model']
        })
        
        if 'linear' in rel['relationships']:
            linear_rel = rel['relationships']['linear']
            data.append({
                'Aspecto': 'Relación Lineal',
                'Valor': linear_rel['equation']
            })
            data.append({
                'Aspecto': 'R² Relación Lineal',
                'Valor': linear_rel['r2']
            })
            
        if 'ratio' in rel['relationships']:
            ratio_rel = rel['relationships']['ratio']
            data.append({
                'Aspecto': 'Ratio Promedio',
                'Valor': ratio_rel['mean']
            })
            data.append({
                'Aspecto': 'Desviación Estándar Ratio',
                'Valor': ratio_rel['std']
            })
            
        return pd.DataFrame(data)
        
    def save_plots(self):
        """Save plots to file"""
        directory = filedialog.askdirectory(title="Select directory to save graphs")
        
        if directory:
            try:
                # Save individual fits
                individual_path = os.path.join(directory, "ajustes_individuales.png")
                self.fig_individual.savefig(individual_path, dpi=300, bbox_inches='tight')
                
                # Save comparison
                comparison_path = os.path.join(directory, "comparacion_y_relacion.png")
                self.fig_comparison.savefig(comparison_path, dpi=300, bbox_inches='tight')
                
                messagebox.showinfo("Success", f"Graphs saved to:\n{directory}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving graphs:\n{str(e)}")
                
    def export_analysis(self):
        """Export detailed analysis report"""
        if not (self.water_fit_results or self.air_fit_results):
            messagebox.showwarning("No Data", "No results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save complete analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("ANÁLISIS COMPLETO - DATOS DE AGUA Y AIRE\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Current analysis summary
                    summary_content = self.summary_text.get(1.0, tk.END)
                    f.write(summary_content)
                    
                    # Detailed results if available
                    if hasattr(self, 'results_text') and self.results_text.get(1.0, tk.END).strip():
                        f.write("\n\nRESULTADOS DETALLADOS:\n")
                        f.write("-" * 30 + "\n")
                        f.write(self.results_text.get(1.0, tk.END))
                        
                messagebox.showinfo("Success", f"Analysis saved to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving analysis:\n{str(e)}")
    
    def export_current_summary(self):
        """Export summary of current experiment with input data, linear fitting, and best fitting R²"""
        if not (self.water_fit_results or self.air_fit_results or self.relationship_results):
            messagebox.showwarning("No Data", "No analysis results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Current Experiment Summary",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    
                    # 1. INPUT DATA SHEET
                    if self.water_data is not None and self.air_data is not None:
                        input_data = {}
                        
                        # Water data
                        water_x_col = self.water_x_col.get()
                        water_y_col = self.water_y_col.get()
                        if water_x_col and water_y_col:
                            input_data['Water_X'] = self.water_data[water_x_col]
                            input_data['Water_Y'] = self.water_data[water_y_col]
                        
                        # Air data
                        air_x_col = self.air_x_col.get()
                        air_y_col = self.air_y_col.get()
                        if air_x_col and air_y_col:
                            input_data['Air_X'] = self.air_data[air_x_col]
                            input_data['Air_Y'] = self.air_data[air_y_col]
                        
                        if input_data:
                            # Ensure all arrays have the same length by padding with NaN
                            max_len = max(len(arr) for arr in input_data.values())
                            for key in input_data:
                                current_len = len(input_data[key])
                                if current_len < max_len:
                                    padding = [np.nan] * (max_len - current_len)
                                    input_data[key] = list(input_data[key]) + padding
                            
                            input_df = pd.DataFrame(input_data)
                            input_df.to_excel(writer, sheet_name='Input_Data', index=False)
                    
                    # 2. FITTING RESULTS SHEET
                    fitting_data = []
                    
                    # Water fitting results
                    if self.water_fit_results:
                        for model_name, results in self.water_fit_results.items():
                            fitting_data.append({
                                'Data_Type': 'Water',
                                'Model': model_name,
                                'R_Squared': results['r2'],
                                'Equation': results.get('equation', 'N/A'),
                                'Is_Best_Fit': '★' if results['r2'] == max(r['r2'] for r in self.water_fit_results.values()) else ''
                            })
                    
                    # Air fitting results
                    if self.air_fit_results:
                        for model_name, results in self.air_fit_results.items():
                            fitting_data.append({
                                'Data_Type': 'Air',
                                'Model': model_name,
                                'R_Squared': results['r2'],
                                'Equation': results.get('equation', 'N/A'),
                                'Is_Best_Fit': '★' if results['r2'] == max(r['r2'] for r in self.air_fit_results.values()) else ''
                            })
                    
                    if fitting_data:
                        fitting_df = pd.DataFrame(fitting_data)
                        fitting_df.to_excel(writer, sheet_name='Curve_Fitting', index=False)
                    
                    # 3. RELATIONSHIPS SHEET
                    if self.relationship_results and 'relationships' in self.relationship_results:
                        relationship_data = []
                        rels = self.relationship_results['relationships']
                        
                        # Find best relationship R²
                        best_r2 = max([r.get('r2', 0) for r in rels.values() if 'r2' in r], default=0)
                        
                        for rel_name, rel_data in rels.items():
                            if 'equation' in rel_data:
                                relationship_data.append({
                                    'Relationship_Type': rel_name.capitalize(),
                                    'Equation': rel_data['equation'],
                                    'R_Squared': rel_data.get('r2', 'N/A'),
                                    'Is_Best': '★' if rel_data.get('r2', 0) == best_r2 else '',
                                    'Type': rel_data.get('type', rel_name)
                                })
                        
                        if relationship_data:
                            # Sort by R² (highest first)
                            relationship_data.sort(key=lambda x: x['R_Squared'] if x['R_Squared'] != 'N/A' else -1, reverse=True)
                            relationship_df = pd.DataFrame(relationship_data)
                            relationship_df.to_excel(writer, sheet_name='Relationships', index=False)
                    
                    # 4. SUMMARY SHEET
                    summary_data = []
                    
                    # General info
                    summary_data.append(['Experiment Analysis Summary', ''])
                    summary_data.append(['Generated on', pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")])
                    summary_data.append(['', ''])
                    
                    # File information
                    if self.water_file_name:
                        summary_data.append(['Water source file', self.water_file_name])
                    if self.air_file_name:
                        summary_data.append(['Air source file', self.air_file_name])
                    summary_data.append(['', ''])
                    
                    # Data points
                    if self.water_data is not None:
                        summary_data.append(['Water data points', len(self.water_data)])
                    if self.air_data is not None:
                        summary_data.append(['Air data points', len(self.air_data)])
                    summary_data.append(['', ''])
                    
                    # Best fits
                    if self.water_fit_results:
                        best_water = max(self.water_fit_results.items(), key=lambda x: x[1]['r2'])
                        summary_data.append(['Best water fit model', best_water[0]])
                        summary_data.append(['Best water fit R²', f"{best_water[1]['r2']:.4f}"])
                        
                        # Linear fit info
                        if 'linear' in self.water_fit_results:
                            water_linear_r2 = self.water_fit_results['linear']['r2']
                            summary_data.append(['Water linear fit R²', f"{water_linear_r2:.4f}"])
                            summary_data.append(['Water linear quality', 'GOOD' if water_linear_r2 > 0.8 else 'ACCEPTABLE' if water_linear_r2 > 0.7 else 'POOR'])
                    
                    if self.air_fit_results:
                        best_air = max(self.air_fit_results.items(), key=lambda x: x[1]['r2'])
                        summary_data.append(['Best air fit model', best_air[0]])
                        summary_data.append(['Best air fit R²', f"{best_air[1]['r2']:.4f}"])
                        
                        # Linear fit info  
                        if 'linear' in self.air_fit_results:
                            air_linear_r2 = self.air_fit_results['linear']['r2']
                            summary_data.append(['Air linear fit R²', f"{air_linear_r2:.4f}"])
                            summary_data.append(['Air linear quality', 'GOOD' if air_linear_r2 > 0.8 else 'ACCEPTABLE' if air_linear_r2 > 0.7 else 'POOR'])
                    
                    summary_data.append(['', ''])
                    
                    # Linear suitability assessment
                    has_good_linear = False
                    if (self.water_fit_results and 'linear' in self.water_fit_results and
                        self.air_fit_results and 'linear' in self.air_fit_results):
                        water_linear_r2 = self.water_fit_results['linear']['r2']
                        air_linear_r2 = self.air_fit_results['linear']['r2']
                        has_good_linear = min(water_linear_r2, air_linear_r2) > 0.7
                        
                        summary_data.append(['Linear analysis suitable', 'YES' if has_good_linear else 'LIMITED'])
                        summary_data.append(['Min linear R²', f"{min(water_linear_r2, air_linear_r2):.4f}"])
                    
                    # Relationships
                    if self.relationship_results and 'relationships' in self.relationship_results:
                        rels = self.relationship_results['relationships']
                        
                        # Linear relationship
                        if 'linear' in rels:
                            summary_data.append(['Linear relationship R²', f"{rels['linear']['r2']:.4f}"])
                            summary_data.append(['Linear equation', rels['linear']['equation']])
                        
                        # Best relationship
                        best_rel = max(
                            [(name, data) for name, data in rels.items() if 'r2' in data],
                            key=lambda x: x[1]['r2'],
                            default=None
                        )
                        if best_rel:
                            summary_data.append(['Best relationship type', best_rel[0]])
                            summary_data.append(['Best relationship R²', f"{best_rel[1]['r2']:.4f}"])
                            summary_data.append(['Best relationship equation', best_rel[1]['equation']])
                    
                    # Average relationship recommendation
                    if has_good_linear:
                        summary_data.append(['', ''])
                        summary_data.append(['Recommendation', 'Suitable for average linear relationship calculation'])
                        summary_data.append(['Next step', 'Save experiment and calculate average with other linear experiments'])
                    
                    summary_df = pd.DataFrame(summary_data, columns=['Parameter', 'Value'])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                messagebox.showinfo("Success", 
                                  f"Current experiment summary exported to:\n{file_path}\n\n"
                                  f"Contents:\n"
                                  f"• Summary: Key results overview\n"
                                  f"• Input_Data: Original water and air data\n"
                                  f"• Curve_Fitting: All curve fitting results with R²\n"
                                  f"• Relationships: All relationship models ranked by R²")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting current summary:\n{str(e)}")
                
    def clear_results(self):
        """Clear all analysis results"""
        self.water_fit_results = None
        self.air_fit_results = None
        self.relationship_results = None
        
        self.summary_text.delete(1.0, tk.END)
        if hasattr(self, 'results_text'):
            self.results_text.delete(1.0, tk.END)
        
        self.fig_individual.clear()
        self.fig_comparison.clear()
        self.canvas_individual.draw()
        self.canvas_comparison.draw()
        
        self.status_var.set("Results cleared")
    
    # Experiment management methods
    def save_experiment(self):
        """Save current experiment to the experiments list"""
        if not (self.water_fit_results or self.air_fit_results or self.relationship_results):
            messagebox.showwarning("No Data", "No analysis results to save")
            return
        
        # Create dialog for experiment naming with automatic suggestion
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Experiment")
        dialog.geometry("550x400")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Main container
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Title
        title_label = tk.Label(main_frame, text="Save Experiment", font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Automatic naming suggestion
        suggested_name = ""
        if self.water_file_name:
            suggested_name = self.water_file_name
            if self.air_file_name and self.air_file_name != self.water_file_name:
                suggested_name += f"_vs_{self.air_file_name}"
        else:
            suggested_name = f"Experiment_{len(self.experiments)+1}"
        
        # Naming options frame
        naming_frame = ttk.LabelFrame(main_frame, text="Experiment Naming Options")
        naming_frame.pack(fill=tk.X, pady=(0, 15))
        
        naming_var = tk.StringVar(value="auto")
        
        # Option 1: Automatic naming based on water file
        auto_frame = ttk.Frame(naming_frame)
        auto_frame.pack(fill=tk.X, padx=15, pady=10)
        
        auto_radio = ttk.Radiobutton(auto_frame, text="Use water file name (automatic):", 
                                    variable=naming_var, value="auto")
        auto_radio.pack(anchor=tk.W)
        
        auto_name_frame = ttk.Frame(auto_frame)
        auto_name_frame.pack(fill=tk.X, padx=(25, 0), pady=(5, 0))
        
        auto_name_label = ttk.Label(auto_name_frame, text=f'"{suggested_name}"', 
                                   foreground="blue", font=('Arial', 10, 'bold'))
        auto_name_label.pack(anchor=tk.W)
        
        # Option 2: Custom naming
        custom_frame = ttk.Frame(naming_frame)
        custom_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        custom_radio = ttk.Radiobutton(custom_frame, text="Custom name:", 
                                      variable=naming_var, value="custom")
        custom_radio.pack(anchor=tk.W)
        
        custom_entry_frame = ttk.Frame(custom_frame)
        custom_entry_frame.pack(fill=tk.X, padx=(25, 0), pady=(5, 0))
        
        custom_entry = ttk.Entry(custom_entry_frame, width=40, font=('Arial', 10))
        custom_entry.pack(fill=tk.X)
        custom_entry.insert(0, suggested_name)
        
        # File information display
        info_frame = ttk.LabelFrame(main_frame, text="Source Files Information")
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_content_frame = ttk.Frame(info_frame)
        info_content_frame.pack(fill=tk.X, padx=15, pady=10)
        
        info_text = f"Water file: {self.water_file_name if self.water_file_name else 'Not loaded'}\n"
        info_text += f"Air file: {self.air_file_name if self.air_file_name else 'Not loaded'}\n"
        info_text += f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        info_label = ttk.Label(info_content_frame, text=info_text, justify=tk.LEFT, font=('Arial', 9))
        info_label.pack(anchor=tk.W)
        
        # Result variable
        result = {'name': None, 'cancelled': True}
        
        def save_with_name():
            if naming_var.get() == "auto":
                name = suggested_name
            else:
                name = custom_entry.get().strip()
            
            if not name:
                messagebox.showwarning("Invalid Name", "Please enter a valid experiment name")
                return
            
            result['name'] = name
            result['cancelled'] = False
            dialog.destroy()
        
        def cancel_save():
            result['cancelled'] = True
            dialog.destroy()
        
        # Enable/disable custom entry based on radio selection
        def on_naming_change():
            if naming_var.get() == "custom":
                custom_entry.config(state='normal')
                custom_entry.focus_set()
                custom_entry.selection_range(0, tk.END)
            else:
                custom_entry.config(state='readonly')
        
        # Bind radio button changes
        naming_var.trace('w', lambda *args: on_naming_change())
        on_naming_change()  # Initial state
        
        # Buttons frame - ensuring it's always visible
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        
        # Create buttons with proper styling
        save_button = ttk.Button(button_frame, text="💾 Save Experiment", command=save_with_name)
        save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_button = ttk.Button(button_frame, text="❌ Cancel", command=cancel_save)
        cancel_button.pack(side=tk.LEFT)
        
        # Make save button default (activated by Enter)
        dialog.bind('<Return>', lambda e: save_with_name())
        dialog.bind('<Escape>', lambda e: cancel_save())
        
        # Focus management
        if naming_var.get() == "auto":
            save_button.focus_set()
        else:
            custom_entry.focus_set()
        
        # Wait for dialog to close
        dialog.wait_window()
        
        if result['cancelled']:
            return
        
        name = result['name']
        
        # Create experiment data with file names and paths
        experiment = {
            'name': name,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'water_data': self.water_data.copy() if self.water_data is not None else None,
            'air_data': self.air_data.copy() if self.air_data is not None else None,
            'water_fit_results': self.water_fit_results.copy() if self.water_fit_results else None,
            'air_fit_results': self.air_fit_results.copy() if self.air_fit_results else None,
            'relationship_results': self.relationship_results.copy() if self.relationship_results else None,
            'water_columns': (self.water_x_col.get(), self.water_y_col.get()),
            'air_columns': (self.air_x_col.get(), self.air_y_col.get()),
            'water_file_name': self.water_file_name,
            'air_file_name': self.air_file_name,
            'water_file_path': getattr(self, 'last_water_file_path', None),
            'air_file_path': getattr(self, 'last_air_file_path', None)
        }
        
        # Check if experiment name already exists
        for i, exp in enumerate(self.experiments):
            if exp['name'] == name:
                if messagebox.askyesno("Overwrite", f"Experiment '{name}' already exists. Overwrite?"):
                    self.experiments[i] = experiment
                    messagebox.showinfo("Success", f"Experiment '{name}' updated")
                    return
                else:
                    return
        
        # Add new experiment
        self.experiments.append(experiment)
        self.current_experiment_name = name
        
        total_experiments = len(self.experiments)
        messagebox.showinfo("Success", 
                          f"Experiment '{name}' saved successfully!\n\n"
                          f"📊 Total experiments: {total_experiments}\n\n"
                          f"💡 Tip: Use 'File' → 'Save as Default Dataset' to save all {total_experiments} experiments\n"
                          f"for automatic loading next time you start the application.")
        self.status_var.set(f"Experiment '{name}' saved ({total_experiments} total)")
    
    def load_experiment(self):
        """Load a previously saved experiment"""
        if not self.experiments:
            messagebox.showwarning("No Experiments", "No saved experiments available")
            return
            
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Experiment")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Select an experiment to load:", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Listbox with experiment details
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        listbox = tk.Listbox(frame, height=15)
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox
        for i, exp in enumerate(self.experiments):
            rel_info = ""
            if exp['relationship_results']:
                if 'linear' in exp['relationship_results'].get('relationships', {}):
                    r2 = exp['relationship_results']['relationships']['linear']['r2']
                    rel_info = f" (Linear R² = {r2:.3f})"
            
            # Add file name info if available
            file_info = ""
            if exp.get('water_file_name'):
                file_info = f" [Water: {exp['water_file_name']}]"
            
            display_text = f"{exp['name']} - {exp['timestamp']}{rel_info}{file_info}"
            listbox.insert(tk.END, display_text)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def load_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select an experiment")
                return
                
            exp = self.experiments[selection[0]]
            self.load_experiment_data(exp)
            dialog.destroy()
        
        def delete_selected():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select an experiment")
                return
                
            exp_name = self.experiments[selection[0]]['name']
            if messagebox.askyesno("Delete Experiment", f"Delete experiment '{exp_name}'?"):
                del self.experiments[selection[0]]
                listbox.delete(selection[0])
                messagebox.showinfo("Success", f"Experiment '{exp_name}' deleted")
        
        tk.Button(button_frame, text="Load", command=load_selected, bg='lightgreen').pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Delete", command=delete_selected, bg='lightcoral').pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def load_experiment_data(self, experiment):
        """Load experiment data into current session"""
        try:
            # Load data
            self.water_data = experiment['water_data']
            self.air_data = experiment['air_data']
            self.water_fit_results = experiment['water_fit_results']
            self.air_fit_results = experiment['air_fit_results']
            self.relationship_results = experiment['relationship_results']
            self.current_experiment_name = experiment['name']
            
            # Load file names if available
            self.water_file_name = experiment.get('water_file_name', None)
            self.air_file_name = experiment.get('air_file_name', None)
            
            # Update file labels
            if hasattr(self, 'water_file_label') and self.water_file_name:
                self.water_file_label.config(text=self.water_file_name)
            if hasattr(self, 'air_file_label') and self.air_file_name:
                self.air_file_label.config(text=self.air_file_name)
            
            # Restore column selections
            if experiment['water_columns']:
                self.water_x_col.set(experiment['water_columns'][0])
                self.water_y_col.set(experiment['water_columns'][1])
            if experiment['air_columns']:
                self.air_x_col.set(experiment['air_columns'][0])
                self.air_y_col.set(experiment['air_columns'][1])
            
            # Update interface
            self.update_data_preview("water")
            self.update_data_preview("air")
            self.update_summary()
            self.update_plots()
            
            messagebox.showinfo("Success", f"Experiment '{experiment['name']}' loaded successfully")
            self.status_var.set(f"Loaded experiment: {experiment['name']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading experiment:\n{str(e)}")
    
    def compare_experiments(self):
        """Compare multiple experiments"""
        if len(self.experiments) < 2:
            messagebox.showwarning("Insufficient Data", "Need at least 2 experiments to compare")
            return
            
        # Create comparison window
        comp_window = tk.Toplevel(self.root)
        comp_window.title("Experiment Comparison")
        comp_window.geometry("1200x800")
        comp_window.transient(self.root)
        
        # Create notebook for different comparison views
        notebook = ttk.Notebook(comp_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary comparison tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary Comparison")
        
        # Create treeview for summary
        columns = ('Name', 'Timestamp', 'Water Model', 'Air Model', 'Linear R²', 'Ratio Mean')
        tree = ttk.Treeview(summary_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Populate comparison data
        for exp in self.experiments:
            water_model = exp['relationship_results']['water_model'] if exp['relationship_results'] else 'N/A'
            air_model = exp['relationship_results']['air_model'] if exp['relationship_results'] else 'N/A'
            
            linear_r2 = 'N/A'
            ratio_mean = 'N/A'
            
            if exp['relationship_results'] and 'relationships' in exp['relationship_results']:
                rels = exp['relationship_results']['relationships']
                if 'linear' in rels:
                    linear_r2 = f"{rels['linear']['r2']:.4f}"
                if 'ratio' in rels:
                    ratio_mean = f"{rels['ratio']['mean']:.4f}"
            
            tree.insert('', tk.END, values=(
                exp['name'], exp['timestamp'], water_model, air_model, linear_r2, ratio_mean
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visual comparison tab
        visual_frame = ttk.Frame(notebook)
        notebook.add(visual_frame, text="Visual Comparison")
        
        # Matplotlib figure for comparison plots
        fig_comp = plt.Figure(figsize=(12, 8))
        canvas_comp = FigureCanvasTkAgg(fig_comp, visual_frame)
        canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot relationships comparison
        ax1 = fig_comp.add_subplot(2, 2, 1)
        ax2 = fig_comp.add_subplot(2, 2, 2)
        ax3 = fig_comp.add_subplot(2, 2, 3)
        ax4 = fig_comp.add_subplot(2, 2, 4)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, exp in enumerate(self.experiments):
            if exp['relationship_results'] and 'relationships' in exp['relationship_results']:
                color = colors[i % len(colors)]
                
                # Plot linear relationships
                if 'linear' in exp['relationship_results']['relationships']:
                    linear = exp['relationship_results']['relationships']['linear']
                    r2 = linear['r2']
                    ax1.bar(i, r2, color=color, alpha=0.7, label=exp['name'])
                
                # Plot ratio means
                if 'ratio' in exp['relationship_results']['relationships']:
                    ratio = exp['relationship_results']['relationships']['ratio']
                    mean_val = ratio['mean']
                    std_val = ratio['std']
                    ax2.errorbar(i, mean_val, yerr=std_val, fmt='o', color=color, 
                               markersize=8, label=exp['name'])
        
        ax1.set_title('Linear Fit R² Comparison')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('R²')
        ax1.set_xticks(range(len(self.experiments)))
        ax1.set_xticklabels([exp['name'] for exp in self.experiments], rotation=45)
        
        ax2.set_title('Ratio Mean Comparison')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Ratio Y_water/Y_air')
        ax2.set_xticks(range(len(self.experiments)))
        ax2.set_xticklabels([exp['name'] for exp in self.experiments], rotation=45)
        
        fig_comp.tight_layout()
        canvas_comp.draw()
    
    def view_experiments(self):
        """View list of all saved experiments"""
        if not self.experiments:
            messagebox.showinfo("No Experiments", "No experiments saved yet")
            return
            
        # Create view window
        view_window = tk.Toplevel(self.root)
        view_window.title("Saved Experiments")
        view_window.geometry("800x600")
        view_window.transient(self.root)
        
        # Create text widget with scrollbar
        frame = tk.Frame(view_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with experiment details
        content = "SAVED EXPERIMENTS SUMMARY\n"
        content += "=" * 50 + "\n\n"
        
        for i, exp in enumerate(self.experiments, 1):
            content += f"{i}. {exp['name']}\n"
            content += f"   Timestamp: {exp['timestamp']}\n"
            
            # File information
            if exp.get('water_file_name') or exp.get('air_file_name'):
                content += f"   Source files - "
                if exp.get('water_file_name'):
                    content += f"Water: {exp['water_file_name']}"
                if exp.get('air_file_name'):
                    if exp.get('water_file_name'):
                        content += f", Air: {exp['air_file_name']}"
                    else:
                        content += f"Air: {exp['air_file_name']}"
                content += "\n"
            
            content += f"   Data points - Water: {len(exp['water_data']) if exp['water_data'] is not None else 0}, "
            content += f"Air: {len(exp['air_data']) if exp['air_data'] is not None else 0}\n"
            
            if exp['relationship_results']:
                content += f"   Models - Water: {exp['relationship_results']['water_model']}, "
                content += f"Air: {exp['relationship_results']['air_model']}\n"
                
                if 'relationships' in exp['relationship_results']:
                    rels = exp['relationship_results']['relationships']
                    if 'linear' in rels:
                        content += f"   Linear relationship: {rels['linear']['equation']} (R² = {rels['linear']['r2']:.4f})\n"
                    if 'ratio' in rels:
                        content += f"   Ratio relationship: {rels['ratio']['equation']}\n"
            
            content += "\n" + "-" * 50 + "\n\n"
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    
    def export_all_experiments(self):
        """Export all experiments to Excel file"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export all experiments",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = []
                    for exp in self.experiments:
                        row = {
                            'Experiment_Name': exp['name'],
                            'Timestamp': exp['timestamp'],
                            'Water_File': exp.get('water_file_name', 'N/A'),
                            'Air_File': exp.get('air_file_name', 'N/A'),
                            'Water_Model': exp['relationship_results']['water_model'] if exp['relationship_results'] else 'N/A',
                            'Air_Model': exp['relationship_results']['air_model'] if exp['relationship_results'] else 'N/A'
                        }
                        
                        if exp['relationship_results'] and 'relationships' in exp['relationship_results']:
                            rels = exp['relationship_results']['relationships']
                            if 'linear' in rels:
                                row['Linear_Equation'] = rels['linear']['equation']
                                row['Linear_R2'] = rels['linear']['r2']
                            if 'ratio' in rels:
                                row['Ratio_Equation'] = rels['ratio']['equation']
                                row['Ratio_Mean'] = rels['ratio']['mean']
                                row['Ratio_Std'] = rels['ratio']['std']
                        
                        summary_data.append(row)
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Individual experiment sheets
                    for exp in self.experiments:
                        sheet_name = exp['name'][:31]  # Excel sheet name limit
                        
                        if exp['water_data'] is not None and exp['air_data'] is not None:
                            # Combine water and air data
                            combined_data = pd.DataFrame({
                                'Water_X': exp['water_data'][exp['water_columns'][0]] if exp['water_columns'][0] else None,
                                'Water_Y': exp['water_data'][exp['water_columns'][1]] if exp['water_columns'][1] else None,
                                'Air_X': exp['air_data'][exp['air_columns'][0]] if exp['air_columns'][0] else None,
                                'Air_Y': exp['air_data'][exp['air_columns'][1]] if exp['air_columns'][1] else None
                            })
                            combined_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                messagebox.showinfo("Success", f"All experiments exported to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting experiments:\n{str(e)}")
    
    def export_summary_report(self):
        """Export comprehensive summary report with input data, linear fitting, and best fitting R²"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Summary Report",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    
                    # 1. SUMMARY OVERVIEW SHEET
                    summary_data = []
                    for exp in self.experiments:
                        # Get best fitting R² for water and air
                        water_best_r2 = 'N/A'
                        air_best_r2 = 'N/A'
                        water_best_model = 'N/A'
                        air_best_model = 'N/A'
                        
                        if exp.get('water_fit_results') and isinstance(exp['water_fit_results'], dict):
                            try:
                                best_water = max(exp['water_fit_results'].items(), key=lambda x: x[1]['r2'])
                                water_best_r2 = f"{best_water[1]['r2']:.4f}"
                                water_best_model = best_water[0]
                            except (ValueError, KeyError, TypeError):
                                pass
                        
                        if exp.get('air_fit_results') and isinstance(exp['air_fit_results'], dict):
                            try:
                                best_air = max(exp['air_fit_results'].items(), key=lambda x: x[1]['r2'])
                                air_best_r2 = f"{best_air[1]['r2']:.4f}"
                                air_best_model = best_air[0]
                            except (ValueError, KeyError, TypeError):
                                pass
                        
                        # Get relationship data
                        linear_r2 = 'N/A'
                        linear_equation = 'N/A'
                        best_relationship_type = 'N/A'
                        best_relationship_r2 = 'N/A'
                        best_relationship_equation = 'N/A'
                        
                        if (exp.get('relationship_results') and 
                            isinstance(exp['relationship_results'], dict) and 
                            'relationships' in exp['relationship_results']):
                            
                            rels = exp['relationship_results']['relationships']
                            if isinstance(rels, dict):
                                # Linear relationship
                                if 'linear' in rels and isinstance(rels['linear'], dict):
                                    if 'r2' in rels['linear']:
                                        linear_r2 = f"{rels['linear']['r2']:.4f}"
                                    if 'equation' in rels['linear']:
                                        linear_equation = rels['linear']['equation']
                                
                                # Find best relationship
                                best_r2 = -1
                                try:
                                    for rel_name, rel_data in rels.items():
                                        if (rel_name != 'ratio' and 
                                            isinstance(rel_data, dict) and 
                                            'r2' in rel_data):
                                            if rel_data['r2'] > best_r2:
                                                best_r2 = rel_data['r2']
                                                best_relationship_type = rel_name
                                                best_relationship_r2 = f"{rel_data['r2']:.4f}"
                                                best_relationship_equation = rel_data.get('equation', 'N/A')
                                except (AttributeError, TypeError, KeyError):
                                    pass
                        
                        # Calculate prediction R² if possible
                        prediction_r2 = 'N/A'
                        air_linear_equation = 'N/A'
                        water_estimated_equation = 'N/A'
                        
                        if (exp.get('water_data') is not None and 
                            exp.get('air_fit_results') and 
                            isinstance(exp['air_fit_results'], dict) and 
                            'linear' in exp['air_fit_results'] and
                            hasattr(self, 'average_relationship') and 
                            self.average_relationship):
                            
                            water_cols = exp.get('water_columns', ('', ''))
                            if water_cols[0] and water_cols[1]:
                                try:
                                    # Get data
                                    water_x_values = exp['water_data'][water_cols[0]].values
                                    water_y_values = exp['water_data'][water_cols[1]].values
                                    
                                    # Get air linear fit parameters
                                    air_linear_fit = exp['air_fit_results']['linear']
                                    if isinstance(air_linear_fit, dict) and 'params' in air_linear_fit:
                                        air_params = air_linear_fit['params']
                                        if len(air_params) >= 2:
                                            # Store equation used for Air Linear
                                            air_linear_equation = f"Air_Linear = {air_params[0]:.6f} × Water_X + {air_params[1]:.6f}"
                                            
                                            # Calculate Air linear values
                                            air_linear_values = air_params[0] * water_x_values + air_params[1]
                                            
                                            # Get average relationship parameters
                                            avg_params = self.average_relationship.get('params', [1, 0])
                                            if len(avg_params) >= 2:
                                                # Store equation used for Water Estimated
                                                water_estimated_equation = f"Water_Estimated = {avg_params[0]:.6f} × Air_Linear + {avg_params[1]:.6f}"
                                                
                                                # Calculate Water estimated
                                                water_estimated = avg_params[0] * air_linear_values + avg_params[1]
                                                
                                                # Calculate prediction R²
                                                prediction_r2 = f"{r2_score(water_y_values, water_estimated):.6f}"
                                except (ValueError, IndexError, KeyError):
                                    prediction_r2 = 'Error'
                        
                        row = {
                            'Experiment_Name': exp['name'],
                            'Timestamp': exp['timestamp'],
                            'Water_Points': len(exp['water_data']) if exp['water_data'] is not None else 0,
                            'Air_Points': len(exp['air_data']) if exp['air_data'] is not None else 0,
                            'Water_Best_Model': water_best_model,
                            'Water_Best_R2': water_best_r2,
                            'Air_Best_Model': air_best_model,
                            'Air_Best_R2': air_best_r2,
                            'Linear_Relationship_R2': linear_r2,
                            'Linear_Relationship_Equation': linear_equation,
                            'Best_Relationship_Type': best_relationship_type,
                            'Best_Relationship_R2': best_relationship_r2,
                            'Best_Relationship_Equation': best_relationship_equation,
                            'Has_Linear_Fit': 'YES' if (exp.get('water_fit_results') and 'linear' in exp['water_fit_results'] and
                                                       exp.get('air_fit_results') and 'linear' in exp['air_fit_results']) else 'NO',
                            'Water_Linear_R2': exp['water_fit_results']['linear']['r2'] if exp.get('water_fit_results') and 'linear' in exp['water_fit_results'] else 'N/A',
                            'Air_Linear_R2': exp['air_fit_results']['linear']['r2'] if exp.get('air_fit_results') and 'linear' in exp['air_fit_results'] else 'N/A',
                            'Has_Air_Linear_Column': 'YES' if (exp.get('water_data') is not None and 
                                                              exp.get('air_fit_results') and 
                                                              isinstance(exp['air_fit_results'], dict) and 
                                                              'linear' in exp['air_fit_results']) else 'NO',
                            'Has_Water_Estimated_Column': 'YES' if (hasattr(self, 'average_relationship') and 
                                                                   self.average_relationship and
                                                                   exp.get('water_data') is not None and 
                                                                   exp.get('air_fit_results') and 
                                                                   isinstance(exp['air_fit_results'], dict) and 
                                                                   'linear' in exp['air_fit_results']) else 'NO',
                            'Average_Relationship_Available': 'YES' if hasattr(self, 'average_relationship') and self.average_relationship else 'NO',
                            'Air_Linear_Equation': air_linear_equation,
                            'Water_Estimated_Equation': water_estimated_equation,
                            'Prediction_R2': prediction_r2,
                            'Prediction_Quality': ('EXCELLENT' if prediction_r2 != 'N/A' and prediction_r2 != 'Error' and float(prediction_r2) > 0.9 else
                                                  'GOOD' if prediction_r2 != 'N/A' and prediction_r2 != 'Error' and float(prediction_r2) > 0.8 else
                                                  'ACCEPTABLE' if prediction_r2 != 'N/A' and prediction_r2 != 'Error' and float(prediction_r2) > 0.7 else
                                                  'POOR' if prediction_r2 != 'N/A' and prediction_r2 != 'Error' else 'N/A'),
                            'Has_Poly2_Water_Estimated': 'YES' if hasattr(self, 'average_poly2_relationship') and self.average_poly2_relationship else 'NO',
                            'Has_Poly3_Water_Estimated': 'YES' if hasattr(self, 'average_poly3_relationship') and self.average_poly3_relationship else 'NO',
                            'Poly2_Relationship_Available': 'YES' if hasattr(self, 'average_poly2_relationship') and self.average_poly2_relationship else 'NO',
                            'Poly3_Relationship_Available': 'YES' if hasattr(self, 'average_poly3_relationship') and self.average_poly3_relationship else 'NO'
                        }
                        summary_data.append(row)
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary_Report', index=False)
                    
                    # 2. DETAILED DATA SHEETS FOR EACH EXPERIMENT
                    for exp in self.experiments:
                        sheet_name = exp['name'][:25]  # Excel sheet name limit
                        
                        # Prepare comprehensive data for this experiment
                        experiment_data = {}
                        
                        # Input data
                        if exp['water_data'] is not None:
                            water_cols = exp.get('water_columns', ('', ''))
                            if water_cols[0] and water_cols[1]:
                                experiment_data['Water_X'] = exp['water_data'][water_cols[0]]
                                experiment_data['Water_Y'] = exp['water_data'][water_cols[1]]
                        
                        if exp['air_data'] is not None:
                            air_cols = exp.get('air_columns', ('', ''))
                            if air_cols[0] and air_cols[1]:
                                experiment_data['Air_X'] = exp['air_data'][air_cols[0]]
                                experiment_data['Air_Y'] = exp['air_data'][air_cols[1]]
                        
                        # Calculate additional columns: Air linear and Water estimated
                        if (exp.get('water_data') is not None and 
                            exp.get('air_fit_results') and 
                            isinstance(exp['air_fit_results'], dict) and 
                            'linear' in exp['air_fit_results']):
                            
                            water_cols = exp.get('water_columns', ('', ''))
                            if water_cols[0] and water_cols[1]:
                                # Get water X values
                                water_x_values = exp['water_data'][water_cols[0]].values
                                
                                # Get air linear fit parameters
                                air_linear_fit = exp['air_fit_results']['linear']
                                if isinstance(air_linear_fit, dict) and 'params' in air_linear_fit:
                                    air_params = air_linear_fit['params']
                                    if len(air_params) >= 2:
                                        # Calculate Air linear: Apply air linear equation to water X values
                                        # Y_air_linear = slope_air * X_water + intercept_air
                                        air_linear_values = air_params[0] * water_x_values + air_params[1]
                                        experiment_data['Air_Linear'] = air_linear_values
                                        
                                        # Store the equation used for Air Linear calculation
                                        air_linear_equation = f"Air_Linear = {air_params[0]:.6f} × Water_X + {air_params[1]:.6f}"
                                        experiment_data['Air_Linear_Equation'] = [air_linear_equation] * len(water_x_values)
                                        
                                        # Calculate Water estimated using average linear relationship
                                        if hasattr(self, 'average_relationship') and self.average_relationship:
                                            avg_params = self.average_relationship.get('params', [1, 0])
                                            if len(avg_params) >= 2:
                                                # Y_water_estimated = avg_slope * Air_linear + avg_intercept
                                                water_estimated = avg_params[0] * air_linear_values + avg_params[1]
                                                experiment_data['Water_Estimated'] = water_estimated
                                                
                                                # Store the equation used for Water Estimated calculation
                                                water_estimated_equation = f"Water_Estimated = {avg_params[0]:.6f} × Air_Linear + {avg_params[1]:.6f}"
                                                experiment_data['Water_Estimated_Equation'] = [water_estimated_equation] * len(water_x_values)
                                                
                                                # Calculate R² between Water Estimated and actual Water Y
                                                water_y_values = exp['water_data'][water_cols[1]].values
                                                try:
                                                    r2_prediction = r2_score(water_y_values, water_estimated)
                                                    experiment_data['Prediction_R2'] = [f"{r2_prediction:.6f}"] * len(water_x_values)
                                                    experiment_data['Prediction_R2_Numeric'] = [r2_prediction] * len(water_x_values)
                                                except (ValueError, IndexError) as e:
                                                    experiment_data['Prediction_R2'] = ["Error: Array length mismatch"] * len(water_x_values)
                                                    experiment_data['Prediction_R2_Numeric'] = [0.0] * len(water_x_values)
                                        
                                        # Calculate additional Water estimated columns using polynomial relationships
                                        # Water Estimated Poly2: Using average polynomial degree 2 relationship
                                        if hasattr(self, 'average_poly2_relationship') and self.average_poly2_relationship:
                                            poly2_params = self.average_poly2_relationship.get('linear_params', [1, 0])
                                            if len(poly2_params) >= 2:
                                                water_estimated_poly2 = poly2_params[0] * air_linear_values + poly2_params[1]
                                                experiment_data['Water_Estimated_Poly2'] = water_estimated_poly2
                                                
                                                poly2_equation = f"Water_Est_Poly2 = {poly2_params[0]:.6f} × Air_Linear + {poly2_params[1]:.6f}"
                                                experiment_data['Water_Estimated_Poly2_Equation'] = [poly2_equation] * len(water_x_values)
                                                
                                                # Calculate R² for poly2 prediction
                                                try:
                                                    r2_poly2 = r2_score(water_y_values, water_estimated_poly2)
                                                    experiment_data['Prediction_R2_Poly2'] = [f"{r2_poly2:.6f}"] * len(water_x_values)
                                                except:
                                                    experiment_data['Prediction_R2_Poly2'] = ["Error"] * len(water_x_values)
                                        
                                        # Water Estimated Poly3: Using average polynomial degree 3 relationship
                                        if hasattr(self, 'average_poly3_relationship') and self.average_poly3_relationship:
                                            poly3_params = self.average_poly3_relationship.get('linear_params', [1, 0])
                                            if len(poly3_params) >= 2:
                                                water_estimated_poly3 = poly3_params[0] * air_linear_values + poly3_params[1]
                                                experiment_data['Water_Estimated_Poly3'] = water_estimated_poly3
                                                
                                                poly3_equation = f"Water_Est_Poly3 = {poly3_params[0]:.6f} × Air_Linear + {poly3_params[1]:.6f}"
                                                experiment_data['Water_Estimated_Poly3_Equation'] = [poly3_equation] * len(water_x_values)
                                                
                                                # Calculate R² for poly3 prediction
                                                try:
                                                    r2_poly3 = r2_score(water_y_values, water_estimated_poly3)
                                                    experiment_data['Prediction_R2_Poly3'] = [f"{r2_poly3:.6f}"] * len(water_x_values)
                                                except:
                                                    experiment_data['Prediction_R2_Poly3'] = ["Error"] * len(water_x_values)
                        
                        # Create DataFrame for this experiment
                        if experiment_data:
                            # Ensure all arrays have the same length by padding with NaN
                            max_len = max(len(arr) for arr in experiment_data.values())
                            for key in experiment_data:
                                current_len = len(experiment_data[key])
                                if current_len < max_len:
                                    # Pad with NaN
                                    padding = [np.nan] * (max_len - current_len)
                                    experiment_data[key] = list(experiment_data[key]) + padding
                            
                            exp_df = pd.DataFrame(experiment_data)
                            exp_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # 3. FITTING RESULTS SHEET
                    fitting_data = []
                    for exp in self.experiments:
                        if exp.get('water_fit_results') and isinstance(exp['water_fit_results'], dict):
                            try:
                                for model_name, results in exp['water_fit_results'].items():
                                    if isinstance(results, dict) and 'r2' in results:
                                        fitting_data.append({
                                            'Experiment': exp['name'],
                                            'Data_Type': 'Water',
                                            'Model': model_name,
                                            'R_Squared': results['r2'],
                                            'Equation': results.get('equation', 'N/A')
                                        })
                            except (AttributeError, TypeError):
                                pass
                        
                        if exp.get('air_fit_results') and isinstance(exp['air_fit_results'], dict):
                            try:
                                for model_name, results in exp['air_fit_results'].items():
                                    if isinstance(results, dict) and 'r2' in results:
                                        fitting_data.append({
                                            'Experiment': exp['name'],
                                            'Data_Type': 'Air',
                                            'Model': model_name,
                                            'R_Squared': results['r2'],
                                            'Equation': results.get('equation', 'N/A')
                                        })
                            except (AttributeError, TypeError):
                                pass
                    
                    if fitting_data:
                        fitting_df = pd.DataFrame(fitting_data)
                        fitting_df.to_excel(writer, sheet_name='Fitting_Results', index=False)
                    
                    # 4. RELATIONSHIPS SHEET
                    relationship_data = []
                    for exp in self.experiments:
                        if (exp.get('relationship_results') and 
                            isinstance(exp['relationship_results'], dict) and 
                            'relationships' in exp['relationship_results']):
                            
                            rels = exp['relationship_results']['relationships']
                            if isinstance(rels, dict):
                                try:
                                    for rel_name, rel_data in rels.items():
                                        if (rel_name != 'ratio' and 
                                            isinstance(rel_data, dict)):  # Skip ratio as it doesn't have R²
                                            relationship_data.append({
                                                'Experiment': exp['name'],
                                                'Relationship_Type': rel_name,
                                                'R_Squared': rel_data.get('r2', 'N/A'),
                                                'Equation': rel_data.get('equation', 'N/A'),
                                                'Is_Best': '★' if (rel_data.get('r2', 0) == max(
                                                    [r.get('r2', 0) for r in rels.values() if isinstance(r, dict) and 'r2' in r], default=0
                                                )) else ''
                                            })
                                except (AttributeError, TypeError, KeyError):
                                    pass
                    
                    if relationship_data:
                        relationship_df = pd.DataFrame(relationship_data)
                        relationship_df.to_excel(writer, sheet_name='Relationships', index=False)
                    
                    # 5. LINEAR ANALYSIS SHEET
                    linear_data = []
                    linear_experiments = []
                    
                    # Filter experiments with linear relationships
                    for exp in self.experiments:
                        if (exp.get('water_fit_results') and 
                            isinstance(exp['water_fit_results'], dict) and 
                            'linear' in exp['water_fit_results'] and
                            exp.get('air_fit_results') and 
                            isinstance(exp['air_fit_results'], dict) and 
                            'linear' in exp['air_fit_results']):
                            
                            linear_experiments.append(exp)
                            
                            water_linear = exp['water_fit_results']['linear']
                            air_linear = exp['air_fit_results']['linear']
                            
                            # Linear relationship data if available
                            linear_rel_r2 = 'N/A'
                            linear_rel_equation = 'N/A'
                            
                            if (exp.get('relationship_results') and 
                                isinstance(exp['relationship_results'], dict) and
                                'relationships' in exp['relationship_results'] and
                                isinstance(exp['relationship_results']['relationships'], dict) and
                                'linear' in exp['relationship_results']['relationships']):
                                
                                linear_rel = exp['relationship_results']['relationships']['linear']
                                if isinstance(linear_rel, dict):
                                    linear_rel_r2 = linear_rel.get('r2', 'N/A')
                                    linear_rel_equation = linear_rel.get('equation', 'N/A')
                            
                            linear_data.append({
                                'Experiment': exp['name'],
                                'Water_Linear_R2': water_linear.get('r2', 'N/A') if isinstance(water_linear, dict) else 'N/A',
                                'Water_Linear_Slope': water_linear.get('params', [0, 0])[0] if isinstance(water_linear, dict) and 'params' in water_linear and len(water_linear['params']) > 0 else 'N/A',
                                'Water_Linear_Intercept': water_linear.get('params', [0, 0])[1] if isinstance(water_linear, dict) and 'params' in water_linear and len(water_linear['params']) > 1 else 'N/A',
                                'Air_Linear_R2': air_linear.get('r2', 'N/A') if isinstance(air_linear, dict) else 'N/A',
                                'Air_Linear_Slope': air_linear.get('params', [0, 0])[0] if isinstance(air_linear, dict) and 'params' in air_linear and len(air_linear['params']) > 0 else 'N/A',
                                'Air_Linear_Intercept': air_linear.get('params', [0, 0])[1] if isinstance(air_linear, dict) and 'params' in air_linear and len(air_linear['params']) > 1 else 'N/A',
                                'Relationship_R2': linear_rel_r2,
                                'Relationship_Equation': linear_rel_equation,
                                'Linear_Quality': ('GOOD' if (isinstance(water_linear, dict) and 
                                                            isinstance(air_linear, dict) and 
                                                            water_linear.get('r2', 0) > 0.8 and 
                                                            air_linear.get('r2', 0) > 0.8) else 
                                                'ACCEPTABLE' if (isinstance(water_linear, dict) and 
                                                               isinstance(air_linear, dict) and 
                                                               water_linear.get('r2', 0) > 0.7 and 
                                                               air_linear.get('r2', 0) > 0.7) else 'POOR')
                            })
                    
                    if linear_data:
                        linear_df = pd.DataFrame(linear_data)
                        linear_df.to_excel(writer, sheet_name='Linear_Analysis', index=False)
                        
                        # 6. AVERAGE LINEAR RELATIONSHIP SHEET (if calculated)
                        if hasattr(self, 'average_relationship') and linear_experiments:
                            avg_rel_data = {
                                'Parameter': [
                                    'Average Linear Equation',
                                    'Slope (a)',
                                    'Intercept (b)',
                                    'Average R²',
                                    'Number of Linear Experiments',
                                    'Quality Rating',
                                    'Usage Formula',
                                    'Expected Accuracy (%)',
                                    'Calculation Date'
                                ],
                                'Value': [
                                    self.average_relationship['equation'],
                                    self.average_relationship['params'][0],
                                    self.average_relationship['params'][1],
                                    self.average_relationship['r2_avg'],
                                    self.average_relationship['n_experiments'],
                                    'EXCELLENT' if self.average_relationship['r2_avg'] > 0.9 else 
                                    'GOOD' if self.average_relationship['r2_avg'] > 0.8 else 
                                    'ACCEPTABLE' if self.average_relationship['r2_avg'] > 0.7 else 'POOR',
                                    f"Y_water = {self.average_relationship['params'][0]:.4f} * Y_air + {self.average_relationship['params'][1]:.4f}",
                                    f"±{(1-self.average_relationship['r2_avg'])*100:.1f}%",
                                    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                                ]
                            }
                            
                            avg_rel_df = pd.DataFrame(avg_rel_data)
                            avg_rel_df.to_excel(writer, sheet_name='Average_Linear_Relationship', index=False)
                            
                            # Individual contributions to average
                            contrib_data = []
                            for i, exp in enumerate(linear_experiments):
                                contrib_data.append({
                                    'Experiment': exp['name'],
                                    'Weight_in_Average': self.average_relationship['weights'][i],
                                    'Individual_R2': self.average_relationship['individual_r2'][i],
                                    'Contribution_Quality': 'HIGH' if self.average_relationship['weights'][i] > np.mean(self.average_relationship['weights']) else 'LOW'
                                })
                            
                            if contrib_data:
                                contrib_df = pd.DataFrame(contrib_data)
                                contrib_df.to_excel(writer, sheet_name='Average_Contributions', index=False)
                    
                    # 7. POLYNOMIAL ANALYSIS SHEETS
                    # Polynomial degree 2 analysis
                    poly2_summary = self.get_polynomial_summary_for_report(2)
                    if poly2_summary:
                        poly2_data = []
                        for rel in poly2_summary['relationships']:
                            poly2_data.append({
                                'Experiment': rel['experiment'],
                                'Water_Poly2_R2': rel['water_r2'],
                                'Air_Poly2_R2': rel['air_r2'],
                                'Linear_Relationship_R2': rel['linear_r2'],
                                'Linear_Slope': rel['linear_slope'],
                                'Linear_Intercept': rel['linear_intercept'],
                                'Linear_Equation': rel['equation'],
                                'Quality': 'EXCELLENT' if rel['linear_r2'] > 0.9 else 
                                          'GOOD' if rel['linear_r2'] > 0.8 else 
                                          'ACCEPTABLE' if rel['linear_r2'] > 0.7 else 'POOR'
                            })
                        
                        poly2_df = pd.DataFrame(poly2_data)
                        poly2_df.to_excel(writer, sheet_name='Polynomial_Degree_2', index=False)
                        
                        # Average polynomial degree 2 relationship
                        avg_poly2_data = {
                            'Parameter': [
                                'Analysis Type',
                                'Average Linear Equation (from Poly-2)',
                                'Average Slope',
                                'Average Intercept',
                                'Average R²',
                                'Number of Experiments',
                                'Quality Rating',
                                'Min R²',
                                'Max R²',
                                'Standard Deviation R²'
                            ],
                            'Value': [
                                'Linear Relationships between Polynomial Degree 2 Curves',
                                poly2_summary['average_equation'],
                                poly2_summary['average_slope'],
                                poly2_summary['average_intercept'],
                                poly2_summary['average_r2'],
                                poly2_summary['n_experiments'],
                                'EXCELLENT' if poly2_summary['average_r2'] > 0.9 else 
                                'GOOD' if poly2_summary['average_r2'] > 0.8 else 
                                'ACCEPTABLE' if poly2_summary['average_r2'] > 0.7 else 'POOR',
                                poly2_summary['min_r2'],
                                poly2_summary['max_r2'],
                                poly2_summary['std_r2']
                            ]
                        }
                        
                        avg_poly2_df = pd.DataFrame(avg_poly2_data)
                        avg_poly2_df.to_excel(writer, sheet_name='Average_Poly2_Relationship', index=False)
                    
                    # Polynomial degree 3 analysis
                    poly3_summary = self.get_polynomial_summary_for_report(3)
                    if poly3_summary:
                        poly3_data = []
                        for rel in poly3_summary['relationships']:
                            poly3_data.append({
                                'Experiment': rel['experiment'],
                                'Water_Poly3_R2': rel['water_r2'],
                                'Air_Poly3_R2': rel['air_r2'],
                                'Linear_Relationship_R2': rel['linear_r2'],
                                'Linear_Slope': rel['linear_slope'],
                                'Linear_Intercept': rel['linear_intercept'],
                                'Linear_Equation': rel['equation'],
                                'Quality': 'EXCELLENT' if rel['linear_r2'] > 0.9 else 
                                          'GOOD' if rel['linear_r2'] > 0.8 else 
                                          'ACCEPTABLE' if rel['linear_r2'] > 0.7 else 'POOR'
                            })
                        
                        poly3_df = pd.DataFrame(poly3_data)
                        poly3_df.to_excel(writer, sheet_name='Polynomial_Degree_3', index=False)
                        
                        # Average polynomial degree 3 relationship
                        avg_poly3_data = {
                            'Parameter': [
                                'Analysis Type',
                                'Average Linear Equation (from Poly-3)',
                                'Average Slope',
                                'Average Intercept',
                                'Average R²',
                                'Number of Experiments',
                                'Quality Rating',
                                'Min R²',
                                'Max R²',
                                'Standard Deviation R²'
                            ],
                            'Value': [
                                'Linear Relationships between Polynomial Degree 3 Curves',
                                poly3_summary['average_equation'],
                                poly3_summary['average_slope'],
                                poly3_summary['average_intercept'],
                                poly3_summary['average_r2'],
                                poly3_summary['n_experiments'],
                                'EXCELLENT' if poly3_summary['average_r2'] > 0.9 else 
                                'GOOD' if poly3_summary['average_r2'] > 0.8 else 
                                'ACCEPTABLE' if poly3_summary['average_r2'] > 0.7 else 'POOR',
                                poly3_summary['min_r2'],
                                poly3_summary['max_r2'],
                                poly3_summary['std_r2']
                            ]
                        }
                        
                        avg_poly3_df = pd.DataFrame(avg_poly3_data)
                        avg_poly3_df.to_excel(writer, sheet_name='Average_Poly3_Relationship', index=False)
                
                # 7. PREDICTION ANALYSIS SHEET
                prediction_data = []
                for exp in self.experiments:
                    if (exp.get('water_data') is not None and 
                        exp.get('air_fit_results') and 
                        isinstance(exp['air_fit_results'], dict) and 
                        'linear' in exp['air_fit_results'] and
                        hasattr(self, 'average_relationship') and 
                        self.average_relationship):
                        
                        water_cols = exp.get('water_columns', ('', ''))
                        if water_cols[0] and water_cols[1]:
                            try:
                                # Get data
                                water_x_values = exp['water_data'][water_cols[0]].values
                                water_y_values = exp['water_data'][water_cols[1]].values
                                
                                # Get air linear fit parameters
                                air_linear_fit = exp['air_fit_results']['linear']
                                if isinstance(air_linear_fit, dict) and 'params' in air_linear_fit:
                                    air_params = air_linear_fit['params']
                                    if len(air_params) >= 2:
                                        # Calculate predictions
                                        air_linear_values = air_params[0] * water_x_values + air_params[1]
                                        
                                        avg_params = self.average_relationship.get('params', [1, 0])
                                        if len(avg_params) >= 2:
                                            water_estimated = avg_params[0] * air_linear_values + avg_params[1]
                                            
                                            # Calculate R² and other metrics
                                            prediction_r2 = r2_score(water_y_values, water_estimated)
                                            mse = np.mean((water_y_values - water_estimated) ** 2)
                                            rmse = np.sqrt(mse)
                                            mae = np.mean(np.abs(water_y_values - water_estimated))
                                            
                                            # Calculate polynomial predictions if available
                                            poly2_r2, poly2_equation, poly2_rmse, poly2_mae = 'N/A', 'N/A', 'N/A', 'N/A'
                                            poly3_r2, poly3_equation, poly3_rmse, poly3_mae = 'N/A', 'N/A', 'N/A', 'N/A'
                                            
                                            if hasattr(self, 'average_poly2_relationship') and self.average_poly2_relationship:
                                                poly2_params = self.average_poly2_relationship.get('linear_params', [1, 0])
                                                if len(poly2_params) >= 2:
                                                    water_estimated_poly2 = poly2_params[0] * air_linear_values + poly2_params[1]
                                                    poly2_r2 = r2_score(water_y_values, water_estimated_poly2)
                                                    poly2_rmse = np.sqrt(np.mean((water_y_values - water_estimated_poly2) ** 2))
                                                    poly2_mae = np.mean(np.abs(water_y_values - water_estimated_poly2))
                                                    poly2_equation = f"Water_Est_Poly2 = {poly2_params[0]:.6f} × Air_Linear + {poly2_params[1]:.6f}"
                                            
                                            if hasattr(self, 'average_poly3_relationship') and self.average_poly3_relationship:
                                                poly3_params = self.average_poly3_relationship.get('linear_params', [1, 0])
                                                if len(poly3_params) >= 2:
                                                    water_estimated_poly3 = poly3_params[0] * air_linear_values + poly3_params[1]
                                                    poly3_r2 = r2_score(water_y_values, water_estimated_poly3)
                                                    poly3_rmse = np.sqrt(np.mean((water_y_values - water_estimated_poly3) ** 2))
                                                    poly3_mae = np.mean(np.abs(water_y_values - water_estimated_poly3))
                                                    poly3_equation = f"Water_Est_Poly3 = {poly3_params[0]:.6f} × Air_Linear + {poly3_params[1]:.6f}"
                                            
                                            prediction_data.append({
                                                'Experiment': exp['name'],
                                                'Air_Linear_Equation': f"Air_Linear = {air_params[0]:.6f} × Water_X + {air_params[1]:.6f}",
                                                'Water_Estimated_Equation': f"Water_Estimated = {avg_params[0]:.6f} × Air_Linear + {avg_params[1]:.6f}",
                                                'Water_Estimated_Poly2_Equation': poly2_equation,
                                                'Water_Estimated_Poly3_Equation': poly3_equation,
                                                'Prediction_R2_Linear': prediction_r2,
                                                'Prediction_R2_Poly2': poly2_r2,
                                                'Prediction_R2_Poly3': poly3_r2,
                                                'RMSE_Linear': rmse,
                                                'RMSE_Poly2': poly2_rmse,
                                                'RMSE_Poly3': poly3_rmse,
                                                'MAE_Linear': mae,
                                                'MAE_Poly2': poly2_mae,
                                                'MAE_Poly3': poly3_mae,
                                                'Data_Points': len(water_x_values),
                                                'Water_Y_Range': f"{np.min(water_y_values):.3f} to {np.max(water_y_values):.3f}",
                                                'Water_Estimated_Range': f"{np.min(water_estimated):.3f} to {np.max(water_estimated):.3f}",
                                                'Air_Linear_Range': f"{np.min(air_linear_values):.3f} to {np.max(air_linear_values):.3f}",
                                                'Quality_Rating_Linear': ('EXCELLENT' if prediction_r2 > 0.9 else
                                                                        'GOOD' if prediction_r2 > 0.8 else
                                                                        'ACCEPTABLE' if prediction_r2 > 0.7 else
                                                                        'POOR'),
                                                'Quality_Rating_Poly2': ('EXCELLENT' if poly2_r2 != 'N/A' and poly2_r2 > 0.9 else
                                                                       'GOOD' if poly2_r2 != 'N/A' and poly2_r2 > 0.8 else
                                                                       'ACCEPTABLE' if poly2_r2 != 'N/A' and poly2_r2 > 0.7 else
                                                                       'POOR' if poly2_r2 != 'N/A' else 'N/A'),
                                                'Quality_Rating_Poly3': ('EXCELLENT' if poly3_r2 != 'N/A' and poly3_r2 > 0.9 else
                                                                       'GOOD' if poly3_r2 != 'N/A' and poly3_r2 > 0.8 else
                                                                       'ACCEPTABLE' if poly3_r2 != 'N/A' and poly3_r2 > 0.7 else
                                                                       'POOR' if poly3_r2 != 'N/A' else 'N/A'),
                                                'Air_Linear_R2': air_linear_fit.get('r2', 'N/A'),
                                                'Average_Relationship_R2': self.average_relationship.get('r2_avg', 'N/A'),
                                                'Poly2_Relationship_R2': self.average_poly2_relationship.get('r2_avg', 'N/A') if hasattr(self, 'average_poly2_relationship') and self.average_poly2_relationship else 'N/A',
                                                'Poly3_Relationship_R2': self.average_poly3_relationship.get('r2_avg', 'N/A') if hasattr(self, 'average_poly3_relationship') and self.average_poly3_relationship else 'N/A'
                                            })
                            except Exception as e:
                                prediction_data.append({
                                    'Experiment': exp['name'],
                                    'Air_Linear_Equation': 'Error',
                                    'Water_Estimated_Equation': 'Error', 
                                    'Prediction_R2': f"Error: {str(e)}",
                                    'RMSE': 'Error',
                                    'MAE': 'Error',
                                    'Data_Points': 'Error',
                                    'Water_Y_Range': 'Error',
                                    'Water_Estimated_Range': 'Error',
                                    'Air_Linear_Range': 'Error',
                                    'Quality_Rating': 'Error',
                                    'Air_Linear_R2': 'Error',
                                    'Average_Relationship_R2': 'Error'
                                })
                
                if prediction_data:
                    prediction_df = pd.DataFrame(prediction_data)
                    prediction_df.to_excel(writer, sheet_name='Prediction_Analysis', index=False)
                
                messagebox.showinfo("Success", 
                                  f"Summary report exported to:\n{file_path}\n\n"
                                  f"Contents:\n"
                                  f"• Summary_Report: Overview of all experiments\n"
                                  f"• Individual sheets: Input data for each experiment\n"
                                  f"  - Water_X, Water_Y: Original water data\n"
                                  f"  - Air_X, Air_Y: Original air data\n"
                                  f"  - Air_Linear: Air pressure drop predicted using air linear fit for water X values\n"
                                  f"  - Water_Estimated: Water pressure drop estimated using average linear relationship\n"
                                  f"  - Water_Estimated_Poly2: Water estimated using polynomial degree 2 linear relationship\n"
                                  f"  - Water_Estimated_Poly3: Water estimated using polynomial degree 3 linear relationship\n"
                                  f"  - Equations: All equations used for calculations\n"
                                  f"  - Prediction_R2: R² values for all prediction methods\n"
                                  f"• Fitting_Results: All curve fitting R² values\n"
                                  f"• Relationships: All relationship models and equations\n"
                                  f"• Linear_Analysis: Linear relationships analysis\n"
                                  f"• Average_Linear_Relationship: Average linear equation\n"
                                  f"• Polynomial_Degree_2: Polynomial degree 2 analysis\n"
                                  f"• Average_Poly2_Relationship: Average poly-2 equation\n"
                                  f"• Polynomial_Degree_3: Polynomial degree 3 analysis\n"
                                  f"• Average_Poly3_Relationship: Average poly-3 equation\n"
                                  f"• Prediction_Analysis: Cross-prediction performance metrics for all methods\n\n"
                                  f"Water estimation columns available:\n"
                                  f"• Water_Estimated (Linear): Y_water = linear_slope × Air_Linear + linear_intercept\n"
                                  f"• Water_Estimated_Poly2: Y_water = poly2_slope × Air_Linear + poly2_intercept\n"
                                  f"• Water_Estimated_Poly3: Y_water = poly3_slope × Air_Linear + poly3_intercept\n"
                                  f"• Air_Linear: Y_air_predicted = air_slope × Water_X + air_intercept\n"
                                  f"• All R² values: Compare performance of different estimation methods")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting summary report:\n{str(e)}")
    
    def test_calculated_columns(self):
        """Test and display calculated columns for verification"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments available to test.")
            return
        
        # Create test window
        test_window = tk.Toplevel(self.root)
        test_window.title("Test Calculated Columns - Air Linear & Water Estimated")
        test_window.geometry("1200x800")
        test_window.transient(self.root)
        
        # Create notebook for different experiments
        notebook = ttk.Notebook(test_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        calculations_found = False
        
        for exp in self.experiments:
            # Check if this experiment can have calculated columns
            if (exp.get('water_data') is not None and 
                exp.get('air_fit_results') and 
                isinstance(exp['air_fit_results'], dict) and 
                'linear' in exp['air_fit_results']):
                
                calculations_found = True
                
                # Create frame for this experiment
                exp_frame = ttk.Frame(notebook)
                notebook.add(exp_frame, text=exp['name'][:20])
                
                # Create text widget for results
                text_widget = tk.Text(exp_frame, wrap=tk.WORD, font=('Courier', 10))
                scrollbar = ttk.Scrollbar(exp_frame, orient=tk.VERTICAL, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Calculate and display results
                content = f"CALCULATED COLUMNS TEST FOR: {exp['name']}\n"
                content += "=" * 80 + "\n\n"
                
                water_cols = exp.get('water_columns', ('', ''))
                if water_cols[0] and water_cols[1]:
                    water_x_values = exp['water_data'][water_cols[0]].values
                    water_y_values = exp['water_data'][water_cols[1]].values
                    
                    content += f"INPUT DATA:\n"
                    content += f"• Water X column: {water_cols[0]}\n"
                    content += f"• Water Y column: {water_cols[1]}\n"
                    content += f"• Number of water points: {len(water_x_values)}\n"
                    content += f"• Water X range: {np.min(water_x_values):.4f} to {np.max(water_x_values):.4f}\n"
                    content += f"• Water Y range: {np.min(water_y_values):.4f} to {np.max(water_y_values):.4f}\n\n"
                    
                    # Air linear fit info
                    air_linear_fit = exp['air_fit_results']['linear']
                    air_params = air_linear_fit['params']
                    air_r2 = air_linear_fit['r2']
                    
                    content += f"AIR LINEAR FIT:\n"
                    content += f"• Air linear R²: {air_r2:.4f}\n"
                    content += f"• Air linear equation: Y_air = {air_params[0]:.4f} × X_air + {air_params[1]:.4f}\n\n"
                    
                    # Calculate Air Linear values
                    air_linear_values = air_params[0] * water_x_values + air_params[1]
                    
                    content += f"CALCULATED AIR LINEAR COLUMN:\n"
                    content += f"• Explanation: Applying air linear equation to water X values\n"
                    content += f"• Formula: Y_air_predicted = {air_params[0]:.4f} × Water_X + {air_params[1]:.4f}\n"
                    content += f"• Purpose: Predict air pressure drop at water flow rates\n"
                    content += f"• Air Linear range: {np.min(air_linear_values):.4f} to {np.max(air_linear_values):.4f}\n"
                    content += f"• First 5 Air Linear predictions:\n"
                    for i in range(min(5, len(air_linear_values))):
                        content += f"  Water_X[{i}]={water_x_values[i]:.4f} → Air_Predicted[{i}]={air_linear_values[i]:.4f}\n"
                    content += "\n"
                    
                    # Water estimated calculation
                    if hasattr(self, 'average_relationship') and self.average_relationship:
                        avg_params = self.average_relationship.get('params', [1, 0])
                        avg_equation = self.average_relationship.get('equation', 'N/A')
                        avg_r2 = self.average_relationship.get('r2_avg', 0)
                        
                        water_estimated = avg_params[0] * air_linear_values + avg_params[1]
                        
                        content += f"AVERAGE LINEAR RELATIONSHIP:\n"
                        content += f"• Equation: {avg_equation}\n"
                        content += f"• Average R²: {avg_r2:.4f}\n"
                        content += f"• Slope: {avg_params[0]:.4f}\n"
                        content += f"• Intercept: {avg_params[1]:.4f}\n\n"
                        
                        content += f"CALCULATED WATER ESTIMATED COLUMN:\n"
                        content += f"• Formula: Y_water = {avg_params[0]:.4f} × Air_Linear + {avg_params[1]:.4f}\n"
                        content += f"• Water Estimated range: {np.min(water_estimated):.4f} to {np.max(water_estimated):.4f}\n"
                        content += f"• First 5 Water Estimated values:\n"
                        for i in range(min(5, len(water_estimated))):
                            content += f"  Air_Linear[{i}]={air_linear_values[i]:.4f} → Water_Estimated[{i}]={water_estimated[i]:.4f}\n"
                        content += "\n"
                        
                        # Comparison with actual water values
                        content += f"COMPARISON WITH ACTUAL WATER VALUES:\n"
                        content += f"• Actual vs Estimated (first 5 points):\n"
                        for i in range(min(5, len(water_y_values))):
                            error = abs(water_y_values[i] - water_estimated[i])
                            rel_error = (error / water_y_values[i]) * 100 if water_y_values[i] != 0 else 0
                            content += f"  Actual={water_y_values[i]:.4f}, Estimated={water_estimated[i]:.4f}, Error={error:.4f} ({rel_error:.1f}%)\n"
                        
                        # Overall statistics
                        errors = np.abs(water_y_values - water_estimated)
                        rel_errors = (errors / water_y_values) * 100
                        content += f"\nOVERALL ACCURACY:\n"
                        content += f"• Mean absolute error: {np.mean(errors):.4f}\n"
                        content += f"• Mean relative error: {np.mean(rel_errors):.1f}%\n"
                        content += f"• Max relative error: {np.max(rel_errors):.1f}%\n"
                        content += f"• R² between actual and estimated: {r2_score(water_y_values, water_estimated):.4f}\n"
                        
                    else:
                        content += f"WATER ESTIMATED COLUMN:\n"
                        content += f"• Cannot calculate: No average linear relationship available\n"
                        content += f"• Please calculate average linear relationship first\n"
                        content += f"• Use: Analysis → Calculate Average Linear Relationship\n"
                    
                    content += f"\n" + "=" * 80 + "\n"
                    content += f"SUMMARY FOR EXPORT:\n"
                    content += f"• This experiment WILL have Air_Linear column in export\n"
                    content += f"• This experiment {'WILL' if hasattr(self, 'average_relationship') and self.average_relationship else 'WILL NOT'} have Water_Estimated column in export\n"
                
                text_widget.insert(tk.END, content)
                text_widget.config(state=tk.DISABLED)
        
        if not calculations_found:
            # No calculations possible
            info_frame = ttk.Frame(notebook)
            notebook.add(info_frame, text="Info")
            
            info_label = tk.Label(info_frame, 
                                text="No experiments found with the required data for calculated columns.\n\n"
                                     "Requirements:\n"
                                     "• Experiment must have water data loaded\n"
                                     "• Experiment must have air linear fit calculated\n\n"
                                     "Please:\n"
                                     "1. Load water and air data\n"
                                     "2. Run curve fitting with linear model\n"
                                     "3. Save the experiment\n"
                                     "4. Calculate average linear relationship (optional, for Water_Estimated column)",
                                justify=tk.LEFT, font=('Arial', 11))
            info_label.pack(expand=True, padx=20, pady=20)
        
        # Close button
        ttk.Button(test_window, text="Close", command=test_window.destroy).pack(pady=10)
    
    def debug_air_linear_calculation(self):
        """Debug function to verify air linear calculation step by step"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments available to debug.")
            return
        
        # Create debug window
        debug_window = tk.Toplevel(self.root)
        debug_window.title("Debug Air Linear Calculation")
        debug_window.geometry("1000x700")
        debug_window.transient(self.root)
        
        # Create text widget
        text_widget = tk.Text(debug_window, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(debug_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        content = "DEBUG: AIR LINEAR CALCULATION\n"
        content += "=" * 60 + "\n\n"
        
        found_experiments = False
        
        for i, exp in enumerate(self.experiments):
            if (exp.get('water_data') is not None and 
                exp.get('air_data') is not None and
                exp.get('air_fit_results') and 
                isinstance(exp['air_fit_results'], dict) and 
                'linear' in exp['air_fit_results']):
                
                found_experiments = True
                
                content += f"EXPERIMENT {i+1}: {exp['name']}\n"
                content += "-" * 40 + "\n"
                
                # Get data info
                water_cols = exp.get('water_columns', ('', ''))
                air_cols = exp.get('air_columns', ('', ''))
                
                content += f"Water columns: X='{water_cols[0]}', Y='{water_cols[1]}'\n"
                content += f"Air columns: X='{air_cols[0]}', Y='{air_cols[1]}'\n\n"
                
                if water_cols[0] and water_cols[1] and air_cols[0] and air_cols[1]:
                    # Get actual data
                    water_x = exp['water_data'][water_cols[0]].values
                    water_y = exp['water_data'][water_cols[1]].values
                    air_x = exp['air_data'][air_cols[0]].values  
                    air_y = exp['air_data'][air_cols[1]].values
                    
                    content += f"Data sizes:\n"
                    content += f"  Water: {len(water_x)} points (X: {np.min(water_x):.3f} to {np.max(water_x):.3f})\n"
                    content += f"  Air: {len(air_x)} points (X: {np.min(air_x):.3f} to {np.max(air_x):.3f})\n\n"
                    
                    # Get air linear fit
                    air_fit = exp['air_fit_results']['linear']
                    air_params = air_fit['params']
                    air_r2 = air_fit['r2']
                    
                    content += f"Air linear fit results:\n"
                    content += f"  R² = {air_r2:.6f}\n"
                    content += f"  Parameters: {air_params}\n"
                    content += f"  Equation: Pressure_Drop = {air_params[0]:.6f} × Flowrate + {air_params[1]:.6f}\n\n"
                    
                    # Verify the air fit with air data
                    air_predicted = air_params[0] * air_x + air_params[1]
                    air_fit_r2_check = r2_score(air_y, air_predicted)
                    
                    content += f"Verification with air data:\n"
                    content += f"  Air fit R² check: {air_fit_r2_check:.6f} (should match {air_r2:.6f})\n"
                    content += f"  First 3 air points:\n"
                    for j in range(min(3, len(air_x))):
                        predicted_val = air_params[0] * air_x[j] + air_params[1]
                        content += f"    X={air_x[j]:.3f}, Y_actual={air_y[j]:.3f}, Y_predicted={predicted_val:.3f}\n"
                    content += "\n"
                    
                    # Now apply to water X values
                    air_linear_values = air_params[0] * water_x + air_params[1]
                    
                    content += f"Applying air equation to water X values:\n"
                    content += f"  Formula: Air_Linear = {air_params[0]:.6f} × Water_X + {air_params[1]:.6f}\n"
                    content += f"  Result range: {np.min(air_linear_values):.3f} to {np.max(air_linear_values):.3f}\n"
                    content += f"  First 5 calculations:\n"
                    for j in range(min(5, len(water_x))):
                        calc = air_params[0] * water_x[j] + air_params[1]
                        content += f"    Water_X[{j}] = {water_x[j]:.3f} → Air_Linear = {air_params[0]:.6f}×{water_x[j]:.3f}+{air_params[1]:.6f} = {calc:.3f}\n"
                    
                    content += f"\n  Manual verification of first calculation:\n"
                    if len(water_x) > 0:
                        manual_calc = air_params[0] * water_x[0] + air_params[1]
                        content += f"    {air_params[0]:.6f} × {water_x[0]:.3f} + {air_params[1]:.6f} = {manual_calc:.6f}\n"
                        content += f"    Array result: {air_linear_values[0]:.6f}\n"
                        content += f"    Match: {'YES' if abs(manual_calc - air_linear_values[0]) < 1e-10 else 'NO'}\n"
                    
                    content += "\n" + "="*60 + "\n\n"
        
        if not found_experiments:
            content += "No experiments found with required data:\n"
            content += "- Must have water data loaded\n"
            content += "- Must have air data loaded\n"
            content += "- Must have air linear fit calculated\n\n"
            
            content += "Current experiments:\n"
            for i, exp in enumerate(self.experiments):
                content += f"{i+1}. {exp['name']}\n"
                content += f"   Water data: {'YES' if exp.get('water_data') is not None else 'NO'}\n"
                content += f"   Air data: {'YES' if exp.get('air_data') is not None else 'NO'}\n"
                content += f"   Air fits: {list(exp.get('air_fit_results', {}).keys()) if exp.get('air_fit_results') else 'None'}\n\n"
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(debug_window, text="Close", command=debug_window.destroy).pack(pady=10)
    
    def save_default_dataset(self):
        """Save all experiments as default dataset"""
        if not self.experiments:
            messagebox.showwarning("No Data", "No experiments to save as default. Please add experiments first.")
            return
        
        try:
            # Prepare all experiments data for saving
            experiments_data = []
            
            for exp in self.experiments:
                exp_data = {
                    'name': exp['name'],
                    'timestamp': exp['timestamp'],
                    'water_file_path': exp.get('water_file_path'),
                    'air_file_path': exp.get('air_file_path'),
                    'water_file_name': exp.get('water_file_name'),
                    'air_file_name': exp.get('air_file_name'),
                    'water_columns': exp.get('water_columns'),
                    'air_columns': exp.get('air_columns'),
                    'has_water_data': exp.get('water_data') is not None,
                    'has_air_data': exp.get('air_data') is not None,
                    'has_water_fits': exp.get('water_fit_results') is not None,
                    'has_air_fits': exp.get('air_fit_results') is not None,
                    'has_relationships': exp.get('relationship_results') is not None
                }
                experiments_data.append(exp_data)
            
            config = {
                'dataset_type': 'full_experiments',
                'total_experiments': len(self.experiments),
                'experiments': experiments_data,
                'has_average_relationship': hasattr(self, 'average_relationship') and self.average_relationship is not None,
                'has_poly2_relationship': hasattr(self, 'average_poly2_relationship') and self.average_poly2_relationship is not None,
                'has_poly3_relationship': hasattr(self, 'average_poly3_relationship') and self.average_poly3_relationship is not None,
                'created_date': pd.Timestamp.now().isoformat(),
                'description': f"Complete dataset with {len(self.experiments)} experiments"
            }
            
            with open(self.default_dataset_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.default_dataset_config = config
            
            # Create summary message
            summary = f"Default dataset saved successfully!\n\n"
            summary += f"📊 Total experiments: {len(self.experiments)}\n\n"
            summary += "Experiments included:\n"
            for i, exp in enumerate(self.experiments, 1):
                water_status = "✓" if exp.get('water_data') is not None else "✗"
                air_status = "✓" if exp.get('air_data') is not None else "✗"
                summary += f"  {i}. {exp['name'][:30]} (Water: {water_status}, Air: {air_status})\n"
            
            summary += f"\n🔄 This complete dataset will be loaded automatically on startup."
            
            messagebox.showinfo("Success", summary)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save default dataset:\n{str(e)}")
    
    def load_default_dataset(self):
        """Load the complete default dataset with all experiments"""
        try:
            import json
            import os
            
            if not os.path.exists(self.default_dataset_file):
                messagebox.showinfo("No Default", "No default dataset found.")
                return
            
            with open(self.default_dataset_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('dataset_type') != 'full_experiments':
                messagebox.showwarning("Incompatible", "This default dataset is from an older version. Please save a new default dataset.")
                return
            
            # Clear current experiments
            self.experiments = []
            
            success_count = 0
            error_messages = []
            
            for exp_config in config.get('experiments', []):
                try:
                    # Create new experiment
                    new_experiment = {
                        'name': exp_config['name'],
                        'timestamp': exp_config['timestamp'],
                        'water_data': None,
                        'air_data': None,
                        'water_fit_results': None,
                        'air_fit_results': None,
                        'relationship_results': None,
                        'water_file_name': exp_config.get('water_file_name'),
                        'air_file_name': exp_config.get('air_file_name'),
                        'water_columns': exp_config.get('water_columns'),
                        'air_columns': exp_config.get('air_columns')
                    }
                    
                    # Load water data if available
                    if exp_config.get('water_file_path') and os.path.exists(exp_config['water_file_path']):
                        try:
                            new_experiment['water_data'] = pd.read_excel(exp_config['water_file_path'])
                            new_experiment['water_file_path'] = exp_config['water_file_path']
                        except Exception as e:
                            error_messages.append(f"⚠️ {exp_config['name']}: Water file error - {str(e)[:50]}")
                    
                    # Load air data if available
                    if exp_config.get('air_file_path') and os.path.exists(exp_config['air_file_path']):
                        try:
                            new_experiment['air_data'] = pd.read_excel(exp_config['air_file_path'])
                            new_experiment['air_file_path'] = exp_config['air_file_path']
                        except Exception as e:
                            error_messages.append(f"⚠️ {exp_config['name']}: Air file error - {str(e)[:50]}")
                    
                    self.experiments.append(new_experiment)
                    success_count += 1
                    
                except Exception as e:
                    error_messages.append(f"❌ {exp_config.get('name', 'Unknown')}: {str(e)[:50]}")
            
            # Update display
            self.update_experiments_list()
            
            # Show results
            message = f"Default Dataset Load Results:\n\n"
            message += f"📊 Total experiments in dataset: {config.get('total_experiments', 0)}\n"
            message += f"✅ Successfully loaded: {success_count}\n"
            message += f"❌ Errors: {len(error_messages)}\n\n"
            
            if success_count > 0:
                message += "Loaded experiments:\n"
                for exp in self.experiments:
                    water_status = "✓" if exp.get('water_data') is not None else "✗"
                    air_status = "✓" if exp.get('air_data') is not None else "✗"
                    message += f"  • {exp['name'][:25]} (W:{water_status} A:{air_status})\n"
                message += "\n"
            
            if error_messages:
                message += "Issues encountered:\n"
                for error in error_messages[:5]:  # Show first 5 errors
                    message += f"  {error}\n"
                if len(error_messages) > 5:
                    message += f"  ... and {len(error_messages) - 5} more\n"
                message += "\n"
            
            message += f"📅 Dataset created: {config.get('created_date', 'Unknown')}\n"
            message += f"📝 {config.get('description', '')}"
            
            if success_count > 0:
                messagebox.showinfo("Default Dataset Loaded", message)
                self.status_var.set(f"Default dataset loaded: {success_count} experiments")
            else:
                messagebox.showwarning("Load Issues", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default dataset:\n{str(e)}")
    
    def clear_default_dataset(self):
        """Clear the default dataset configuration"""
        try:
            import json
            import os
            
            # Check if file exists and get info before deleting
            if os.path.exists(self.default_dataset_file):
                try:
                    with open(self.default_dataset_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    dataset_info = f"Dataset contains {config.get('total_experiments', 0)} experiments"
                    if config.get('experiments'):
                        dataset_info += ":\n"
                        for exp in config['experiments'][:5]:  # Show first 5
                            dataset_info += f"  • {exp.get('name', 'Unknown')}\n"
                        if len(config['experiments']) > 5:
                            dataset_info += f"  ... and {len(config['experiments']) - 5} more\n"
                except:
                    dataset_info = "Default dataset file found"
                
                # Confirm deletion
                confirm = messagebox.askyesno("Confirm Clear", 
                                            f"Clear default dataset configuration?\n\n"
                                            f"{dataset_info}\n\n"
                                            f"This will remove the auto-load configuration but won't\n"
                                            f"delete your original Excel files.")
                
                if confirm:
                    os.remove(self.default_dataset_file)
                    self.default_dataset_config = None
                    messagebox.showinfo("Success", "Default dataset configuration cleared.\n\nThe application will no longer auto-load experiments on startup.")
            else:
                messagebox.showinfo("No Default", "No default dataset configuration found.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear default dataset:\n{str(e)}")
    
    def auto_load_default_dataset(self):
        """Automatically load default dataset on startup (silent)"""
        try:
            import json
            import os
            
            if not os.path.exists(self.default_dataset_file):
                return
            
            with open(self.default_dataset_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('dataset_type') != 'full_experiments':
                return  # Skip old format
            
            # Clear current experiments
            self.experiments = []
            
            success_count = 0
            
            for exp_config in config.get('experiments', []):
                try:
                    # Create new experiment
                    new_experiment = {
                        'name': exp_config['name'],
                        'timestamp': exp_config['timestamp'],
                        'water_data': None,
                        'air_data': None,
                        'water_fit_results': None,
                        'air_fit_results': None,
                        'relationship_results': None,
                        'water_file_name': exp_config.get('water_file_name'),
                        'air_file_name': exp_config.get('air_file_name'),
                        'water_columns': exp_config.get('water_columns'),
                        'air_columns': exp_config.get('air_columns')
                    }
                    
                    # Load water data silently
                    if exp_config.get('water_file_path') and os.path.exists(exp_config['water_file_path']):
                        try:
                            new_experiment['water_data'] = pd.read_excel(exp_config['water_file_path'])
                            new_experiment['water_file_path'] = exp_config['water_file_path']
                        except:
                            pass  # Silent failure
                    
                    # Load air data silently
                    if exp_config.get('air_file_path') and os.path.exists(exp_config['air_file_path']):
                        try:
                            new_experiment['air_data'] = pd.read_excel(exp_config['air_file_path'])
                            new_experiment['air_file_path'] = exp_config['air_file_path']
                        except:
                            pass  # Silent failure
                    
                    self.experiments.append(new_experiment)
                    success_count += 1
                    
                except:
                    pass  # Silent failure
            
            if success_count > 0:
                self.update_experiments_list()
                self.status_var.set(f"Auto-loaded default dataset: {success_count} experiments")
                print(f"Default dataset auto-loaded: {success_count} experiments from {config.get('total_experiments', 0)} total")
                
        except:
            pass  # Silent failure on startup
    
    def update_experiments_list(self):
        """Update the experiments list display"""
        # This function should update any GUI elements that show the experiments list
        if hasattr(self, 'experiments_listbox'):
            self.experiments_listbox.delete(0, tk.END)
            for exp in self.experiments:
                water_status = "W✓" if exp.get('water_data') is not None else "W✗"
                air_status = "A✓" if exp.get('air_data') is not None else "A✗"
                display_text = f"{exp['name']} ({water_status} {air_status})"
                self.experiments_listbox.insert(tk.END, display_text)
    
    def update_water_column_options(self):
        """Update water column options after loading data"""
        if self.water_data is not None:
            columns = self.water_data.columns.tolist()
            self.water_x_combo['values'] = columns
            self.water_y_combo['values'] = columns
    
    def update_air_column_options(self):
        """Update air column options after loading data"""
        if self.air_data is not None:
            columns = self.air_data.columns.tolist()
            self.air_x_combo['values'] = columns
            self.air_y_combo['values'] = columns
    
    def update_status_display(self):
        """Update status display after loading data"""
        water_status = "✓ Water" if self.water_data is not None else "✗ Water"
        air_status = "✓ Air" if self.air_data is not None else "✗ Air"
        self.status_var.set(f"Data loaded: {water_status}, {air_status}")
        
        # Update file labels if they exist
        if hasattr(self, 'water_file_label') and self.water_file_name:
            self.water_file_label.config(text=self.water_file_name)
        if hasattr(self, 'air_file_label') and self.air_file_name:
            self.air_file_label.config(text=self.air_file_name)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
                           "Water and Air Data Analyzer\n"
                           "Version 1.0 - Standalone\n\n"
                           "Tool for point cloud analysis\n"
                           "and finding relationships between water and air data.\n\n"
                           "Author: Lucía Ayllón")


def main():
    """Main function to run the standalone application"""
    try:
        print("Iniciando Analizador de Datos de Agua y Aire...")
        print("=" * 50)
        print("Aplicación standalone - Todas las dependencias incluidas")
        print("=" * 50)
        
        root = tk.Tk()
        app = WaterAirAnalyzer(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"Error de importación: {e}")
        print("\nPor favor instala las dependencias requeridas:")
        print("pip install numpy matplotlib pandas scipy scikit-learn openpyxl")
        return 1
        
    except Exception as e:
        print(f"Error en la aplicación: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
