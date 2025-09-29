#!/usr/bin/env python3
"""
Magnetocaloric Heat Exchanger Pressure Drop Test System
Communicates with Bronkhorst FLEXI-FLOW Compact mass flow controller
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import serial
import threading
import time
import json
from datetime import datetime
import os
import queue

class BronkhorstController:
    """Communication interface for Bronkhorst FLEXI-FLOW Compact"""
    
    def __init__(self, port='COM3', baudrate=38400, address=1):
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.connection = None
        self.connected = False
        self.simulation_mode = True  # Set to False for real hardware
        self.modbus_controller = None  # Initialize to None
        
        # Thread-safe data storage
        self.latest_data = None
        self.data_lock = threading.Lock()
        
        # Command lock to prevent simultaneous Modbus operations
        self.command_lock = threading.Lock()
        
        # For real hardware, import and use the Modbus controller
        if not self.simulation_mode:
            try:
                from modbus_controller import BronkhorstModbusController
                self.modbus_controller = BronkhorstModbusController(port, address, baudrate)
            except ImportError:
                print("Modbus controller not available, using simulation mode")
                self.simulation_mode = True
        
    def connect(self):
        """Establish connection to the flow controller"""
        if self.simulation_mode:
            # Simulate connection for development/testing
            self.connected = True
            print(f"Simulated connection to {self.port}")
            return True
        else:
            # Use real Modbus connection
            self.connected = self.modbus_controller.connect()
            return self.connected
    
    def disconnect(self):
        """Close connection"""
        if self.simulation_mode:
            self.connected = False
        else:
            if self.modbus_controller is not None:
                self.modbus_controller.disconnect()
            self.connected = False
    
    def set_flow_rate(self, flow_rate):
        """Set flow rate in ml/min"""
        if self.simulation_mode:
            print(f"Simulated: Setting flow rate to {flow_rate} ml/min")
            return True
        else:
            with self.command_lock:
                time.sleep(0.1)  # Small delay between commands
                return self.modbus_controller.set_flow_rate(flow_rate)
    
    def get_measurements(self):
        """Get current measurements: flow, P1 (inlet), P2 (outlet), temperature"""
        
        if self.simulation_mode:
            # Simulate realistic values for development
            flow = np.random.normal(1000, 50)  # ml/min
            p1 = np.random.normal(2.5, 0.1)    # bar(a) - inlet pressure
            p2 = 1.013  # bar(a) - atmospheric outlet
            temp = np.random.normal(23, 1)     # °C
            
            return {
                'flow_rate': max(0, flow),
                'inlet_pressure': max(0, p1),
                'outlet_pressure': p2,
                'temperature': temp,
                'pressure_drop': max(0, p1 - p2),
                'timestamp': time.time()
            }
        else:
            # Use real Modbus measurements with locking
            with self.command_lock:
                time.sleep(0.1)  # Small delay between commands
                return self.modbus_controller.get_all_measurements()
    
    def get_latest_data(self):
        """Get the latest cached data (thread-safe)"""
        with self.data_lock:
            return self.latest_data
    
    def update_latest_data(self, data):
        """Update the latest cached data (thread-safe)"""
        with self.data_lock:
            self.latest_data = data

class PressureDropTester:
    """Main application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Magnetocaloric Pressure Drop Tester")
        self.root.geometry("1200x800")
        
        # Data storage
        self.current_data = []
        self.reference_data = []
        self.test_running = False
        self.test_thread = None
        
        # Background reading thread
        self.reading_thread = None
        self.reading_active = False
        self.update_scheduled = False  # Prevent multiple update_readings schedules
        
        # Controller
        self.controller = BronkhorstController()
        
        # Create GUI
        self.create_gui()
        
        # Test parameters
        self.flow_rates = []
        self.current_sample = ""
        
    def create_gui(self):
        """Create the main GUI interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="Controller Connection", padding="5")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=0, padx=(0, 5))
        self.port_var = tk.StringVar(value="COM1")
        ttk.Combobox(conn_frame, textvariable=self.port_var, 
                    values=["COM1", "COM2", "COM3", "COM4", "COM5"]).grid(row=0, column=1, padx=(0, 10))
        
        # Simulation mode toggle
        self.sim_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(conn_frame, text="Simulation Mode", 
                       variable=self.sim_mode_var, 
                       command=self.toggle_simulation).grid(row=0, column=2, padx=(10, 10))
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=3)
        
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.grid(row=0, column=4, padx=(10, 0))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Test Control", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Sample name
        ttk.Label(control_frame, text="Sample Name:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.sample_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.sample_var, width=20).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Manual flow control
        ttk.Label(control_frame, text="Manual Flow (ml/min):").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.manual_flow_var = tk.StringVar(value="1000")
        ttk.Entry(control_frame, textvariable=self.manual_flow_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="Set Flow", command=self.set_manual_flow).grid(row=1, column=2, padx=(5, 0), pady=(0, 5))
        
        # Auto test parameters
        ttk.Label(control_frame, text="Auto Test Range:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        flow_frame = ttk.Frame(control_frame)
        flow_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(flow_frame, text="Start:").grid(row=0, column=0)
        self.start_flow_var = tk.StringVar(value="500")
        ttk.Entry(flow_frame, textvariable=self.start_flow_var, width=8).grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(flow_frame, text="End:").grid(row=0, column=2)
        self.end_flow_var = tk.StringVar(value="4000")
        ttk.Entry(flow_frame, textvariable=self.end_flow_var, width=8).grid(row=0, column=3, padx=(5, 10))
        
        ttk.Label(flow_frame, text="Steps:").grid(row=0, column=4)
        self.steps_var = tk.StringVar(value="10")
        ttk.Entry(flow_frame, textvariable=self.steps_var, width=8).grid(row=0, column=5, padx=(5, 0))
        
        # Test buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=(10, 0))
        
        self.ref_btn = ttk.Button(button_frame, text="Record Reference", command=self.record_reference)
        self.ref_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.test_btn = ttk.Button(button_frame, text="Start Test", command=self.start_test)
        self.test_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_test, state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=(0, 5))
        
        # Data buttons
        data_frame = ttk.Frame(control_frame)
        data_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(data_frame, text="Save Data", command=self.save_data).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(data_frame, text="Load Reference", command=self.load_reference).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(data_frame, text="Clear Plot", command=self.clear_plot).grid(row=0, column=2)
        
        # Current readings frame
        readings_frame = ttk.LabelFrame(main_frame, text="Current Readings", padding="5")
        readings_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N), padx=(0, 0))
        
        self.flow_label = ttk.Label(readings_frame, text="Flow: --- ml/min")
        self.flow_label.grid(row=0, column=0, sticky=tk.W)
        
        self.p1_label = ttk.Label(readings_frame, text="Inlet P: --- bar")
        self.p1_label.grid(row=1, column=0, sticky=tk.W)
        
        self.p2_label = ttk.Label(readings_frame, text="Outlet P: --- bar")
        self.p2_label.grid(row=2, column=0, sticky=tk.W)
        
        self.dp_label = ttk.Label(readings_frame, text="ΔP: --- mbar")
        self.dp_label.grid(row=3, column=0, sticky=tk.W)
        
        self.temp_label = ttk.Label(readings_frame, text="Temp: --- °C")
        self.temp_label.grid(row=4, column=0, sticky=tk.W)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Pressure Drop vs Flow Rate", padding="5")
        plot_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel('Flow Rate (ml/min)')
        self.ax.set_ylabel('Pressure Drop (mbar)')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start reading updates
        self.update_scheduled = True
        self.update_readings()
    
    def start_background_reading(self):
        """Start background thread for reading measurements"""
        if self.reading_thread is None or not self.reading_thread.is_alive():
            self.reading_active = True
            self.reading_thread = threading.Thread(target=self._background_reading_loop, daemon=True)
            self.reading_thread.start()
    
    def stop_background_reading(self):
        """Stop background reading thread"""
        self.reading_active = False
        if self.reading_thread:
            self.reading_thread.join(timeout=2.0)
    
    def _background_reading_loop(self):
        """Background loop to continuously read measurements"""
        while self.reading_active and self.controller.connected:
            try:
                data = self.controller.get_measurements()
                if data:
                    self.controller.update_latest_data(data)
            except Exception as e:
                print(f"Error reading measurements: {e}")
            time.sleep(2.0)  # Increased delay - read every 2 seconds to avoid overwhelming device
        
    def toggle_simulation(self):
        """Toggle between simulation and real hardware mode"""
        new_mode = self.sim_mode_var.get()
        
        # If switching to hardware mode, try to import modbus controller
        if not new_mode and self.controller.modbus_controller is None:
            try:
                from modbus_controller import BronkhorstModbusController
                self.controller.modbus_controller = BronkhorstModbusController(
                    self.controller.port, 
                    self.controller.address, 
                    self.controller.baudrate
                )
            except ImportError:
                messagebox.showerror("Error", "Modbus controller not available. Staying in simulation mode.")
                self.sim_mode_var.set(True)
                return
        
        self.controller.simulation_mode = new_mode
        
        if self.controller.connected:
            # Reconnect with new mode
            self.stop_background_reading()
            self.controller.disconnect()
            self.status_label.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect")
    
    def toggle_connection(self):
        """Connect or disconnect from controller"""
        if not self.controller.connected:
            self.controller.port = self.port_var.get()
            # Disable button during connection attempt
            self.connect_btn.config(state='disabled')
            self.status_label.config(text="Connecting...", foreground="orange")
            
            # Connect in background thread to avoid blocking
            def connect_thread():
                success = self.controller.connect()
                if success:
                    self.start_background_reading()
                    self.root.after(0, lambda: self.status_label.config(text="Connected", foreground="green"))
                    self.root.after(0, lambda: self.connect_btn.config(text="Disconnect", state='normal'))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to connect to controller"))
                    self.root.after(0, lambda: self.status_label.config(text="Disconnected", foreground="red"))
                    self.root.after(0, lambda: self.connect_btn.config(text="Connect", state='normal'))
            
            threading.Thread(target=connect_thread, daemon=True).start()
        else:
            self.stop_background_reading()
            self.controller.disconnect()
            self.status_label.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect")
    
    def update_readings(self):
        """Update current readings display"""
        if not self.update_scheduled:
            return  # Don't continue if we've been cancelled
            
        if self.controller.connected:
            data = self.controller.get_latest_data()
            if data:
                try:
                    self.flow_label.config(text=f"Flow: {data['flow_rate']:.1f} ml/min")
                    self.p1_label.config(text=f"Inlet P: {data['inlet_pressure']:.3f} bar")
                    self.p2_label.config(text=f"Outlet P: {data['outlet_pressure']:.3f} bar")
                    self.dp_label.config(text=f"ΔP: {data['pressure_drop']*1000:.1f} mbar")
                    self.temp_label.config(text=f"Temp: {data['temperature']:.1f} °C")
                except tk.TclError:
                    # Widget was destroyed, stop updates
                    self.update_scheduled = False
                    return
        
        # Schedule next update
        if self.update_scheduled:
            self.root.after(1000, self.update_readings)
    
    def set_manual_flow(self):
        """Set manual flow rate"""
        try:
            flow_rate = float(self.manual_flow_var.get())
            if 0 <= flow_rate <= 5000:
                # Run in background thread to avoid blocking
                threading.Thread(target=lambda: self.controller.set_flow_rate(flow_rate), daemon=True).start()
            else:
                messagebox.showerror("Error", "Flow rate must be between 0 and 5000 ml/min")
        except ValueError:
            messagebox.showerror("Error", "Invalid flow rate value")
    
    def record_reference(self):
        """Record reference pressure drop (system without block)"""
        if not self.controller.connected:
            messagebox.showerror("Error", "Controller not connected")
            return
        
        if not self.sample_var.get().strip():
            messagebox.showerror("Error", "Please enter a sample name")
            return
        
        # Create flow rate series
        try:
            start = float(self.start_flow_var.get())
            end = float(self.end_flow_var.get())
            steps = int(self.steps_var.get())
            flow_rates = np.linspace(start, end, steps)
        except ValueError:
            messagebox.showerror("Error", "Invalid test parameters")
            return
        
        # Start reference measurement in separate thread
        self.test_running = True
        self.ref_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        def run_reference():
            self.reference_data = []
            
            for flow_rate in flow_rates:
                if not self.test_running:
                    break
                
                # Set flow rate and wait for stabilization
                self.controller.set_flow_rate(flow_rate)
                time.sleep(3)  # Wait for stabilization
                
                # Take multiple measurements and average
                measurements = []
                for _ in range(5):
                    data = self.controller.get_measurements()
                    if data:
                        measurements.append(data)
                    time.sleep(0.5)
                
                if measurements:
                    avg_data = {
                        'flow_rate': np.mean([m['flow_rate'] for m in measurements]),
                        'pressure_drop': np.mean([m['pressure_drop'] for m in measurements]),
                        'inlet_pressure': np.mean([m['inlet_pressure'] for m in measurements]),
                        'temperature': np.mean([m['temperature'] for m in measurements])
                    }
                    self.reference_data.append(avg_data)
            
            # Reset UI
            self.test_running = False
            self.root.after(0, lambda: self.ref_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            self.root.after(0, lambda: messagebox.showinfo("Complete", "Reference measurement completed"))
        
        self.test_thread = threading.Thread(target=run_reference, daemon=True)
        self.test_thread.start()
    
    def start_test(self):
        """Start automated pressure drop test"""
        if not self.controller.connected:
            messagebox.showerror("Error", "Controller not connected")
            return
        
        if not self.sample_var.get().strip():
            messagebox.showerror("Error", "Please enter a sample name")
            return
        
        # Create flow rate series
        try:
            start = float(self.start_flow_var.get())
            end = float(self.end_flow_var.get())
            steps = int(self.steps_var.get())
            flow_rates = np.linspace(start, end, steps)
        except ValueError:
            messagebox.showerror("Error", "Invalid test parameters")
            return
        
        # Start test in separate thread
        self.test_running = True
        self.test_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.current_sample = self.sample_var.get()
        
        def run_test():
            self.current_data = []
            
            for flow_rate in flow_rates:
                if not self.test_running:
                    break
                
                # Set flow rate and wait for stabilization
                self.controller.set_flow_rate(flow_rate)
                time.sleep(3)  # Wait for stabilization
                
                # Take multiple measurements and average
                measurements = []
                for _ in range(5):
                    data = self.controller.get_measurements()
                    if data:
                        measurements.append(data)
                    time.sleep(0.5)
                
                if measurements:
                    avg_data = {
                        'flow_rate': np.mean([m['flow_rate'] for m in measurements]),
                        'pressure_drop_raw': np.mean([m['pressure_drop'] for m in measurements]),
                        'inlet_pressure': np.mean([m['inlet_pressure'] for m in measurements]),
                        'temperature': np.mean([m['temperature'] for m in measurements]),
                        'sample': self.current_sample
                    }
                    
                    # Apply reference compensation if available
                    if self.reference_data:
                        ref_dp = self.interpolate_reference(avg_data['flow_rate'])
                        avg_data['pressure_drop_compensated'] = avg_data['pressure_drop_raw'] - ref_dp
                    else:
                        avg_data['pressure_drop_compensated'] = avg_data['pressure_drop_raw']
                    
                    self.current_data.append(avg_data)
                    
                    # Update plot
                    self.root.after(0, self.update_plot)
            
            # Reset UI
            self.test_running = False
            self.root.after(0, lambda: self.test_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            self.root.after(0, lambda: messagebox.showinfo("Complete", "Test completed"))
        
        self.test_thread = threading.Thread(target=run_test, daemon=True)
        self.test_thread.start()
    
    def stop_test(self):
        """Stop current test"""
        self.test_running = False
        self.test_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def interpolate_reference(self, flow_rate):
        """Interpolate reference pressure drop for given flow rate"""
        if not self.reference_data:
            return 0
        
        ref_flows = [data['flow_rate'] for data in self.reference_data]
        ref_pressures = [data['pressure_drop'] for data in self.reference_data]
        
        return np.interp(flow_rate, ref_flows, ref_pressures)
    
    def update_plot(self):
        """Update the plot with current data"""
        self.ax.clear()
        self.ax.set_xlabel('Flow Rate (ml/min)')
        self.ax.set_ylabel('Pressure Drop (mbar)')
        self.ax.grid(True, alpha=0.3)
        
        # Plot reference data
        if self.reference_data:
            ref_flows = [data['flow_rate'] for data in self.reference_data]
            ref_pressures = [data['pressure_drop'] * 1000 for data in self.reference_data]  # Convert to mbar
            self.ax.plot(ref_flows, ref_pressures, 'b--', label='Reference (no block)', linewidth=2)
        
        # Plot current test data
        if self.current_data:
            flows = [data['flow_rate'] for data in self.current_data]
            pressures_raw = [data['pressure_drop_raw'] * 1000 for data in self.current_data]
            pressures_comp = [data['pressure_drop_compensated'] * 1000 for data in self.current_data]
            
            self.ax.plot(flows, pressures_raw, 'r-o', label=f'{self.current_sample} (raw)', linewidth=2, markersize=4)
            self.ax.plot(flows, pressures_comp, 'g-s', label=f'{self.current_sample} (compensated)', linewidth=2, markersize=4)
        
        self.ax.legend()
        self.canvas.draw()
    
    def clear_plot(self):
        """Clear the plot"""
        self.current_data = []
        self.update_plot()
    
    def save_data(self):
        """Save current data to CSV file"""
        if not self.current_data:
            messagebox.showwarning("Warning", "No data to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialname=f"{self.current_sample}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if filename:
            df = pd.DataFrame(self.current_data)
            df.to_csv(filename, index=False)
            
            # Also save reference data if available
            if self.reference_data:
                ref_filename = filename.replace('.csv', '_reference.csv')
                ref_df = pd.DataFrame(self.reference_data)
                ref_df.to_csv(ref_filename, index=False)
            
            messagebox.showinfo("Success", f"Data saved to {filename}")
    
    def load_reference(self):
        """Load reference data from CSV file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.read_csv(filename)
                self.reference_data = df.to_dict('records')
                self.update_plot()
                messagebox.showinfo("Success", "Reference data loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load reference data: {e}")
    
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        finally:
            # Clean up on exit
            self.update_scheduled = False
            self.stop_background_reading()
            if self.controller.connected:
                self.controller.disconnect()

if __name__ == "__main__":
    app = PressureDropTester()
    app.run()