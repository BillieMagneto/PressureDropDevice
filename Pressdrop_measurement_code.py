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
import threading
import time
import json
from datetime import datetime
import os
import logging
import propar

# Set up logging to both file and console
log_filename = 'pressure_drop_measurements.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# These lines reduce unnecessary warning messages from the graphical components
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)

logging.info("Starting Pressure Drop Test Application")

# A class is like a blueprint for creating objects. Think of it as a template that defines
# what something can do and what information it can store. For example, if you were designing
# a car system, you might have a 'Car' class that defines what all cars have in common.

class BronkhorstController:
    """This class handles all communication with the Bronkhorst flow controller device
    using the official bronkhorst-propar Python driver."""
    
    def __init__(self, port=None, baudrate=38400):
        """Initialize the controller
        
        Args:
            port (str): COM port name (e.g., 'COM1')
            baudrate (int): Communication speed (default 38400 for Bronkhorst)
        """
        self.port = port          
        self.baudrate = baudrate  
        self.simulation_mode = False
        self.connected = False
        self.instrument = None    # propar instrument instance
        self.master = None       # propar master instance
        self.p_atm = None   # baseline atmospheric pressure (bar[a]) measured at no-flow
        
    def connect(self):
        """Establish connection to the flow controller using the Propar protocol
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.simulation_mode:
            self.connected = True
            print(f"Simulated connection to {self.port}")
            return True
            
        try:
            # First, check if port exists and is available
            import serial.tools.list_ports
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            if self.port not in available_ports:
                logging.error(f"Port {self.port} not found. Available ports: {available_ports}")
                return False

            # Create a new Propar instrument instance
            self.instrument = propar.instrument(self.port, baudrate=self.baudrate)
            
            try:
                # Just verify we can read the measure value (parameter 8)
                measure = self.instrument.readParameter(8)
                logging.info(f"Initial measure value: {measure}")
                
                self.connected = True
                logging.info(f"Successfully connected to flow controller on {self.port}")
                return True
            except Exception as e:
                logging.error(f"Could not verify device connection: {str(e)}")
                return False
            
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            self.connected = False
            self.master = None
            self.instrument = None
            return False
                
        except Exception as e:
            logging.error(f"Error connecting to device: {str(e)}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Close the connection to the flow controller"""
        if self.simulation_mode:
            self.connected = False
            return
            
        try:
            if self.master:
                self.master.close()
            self.master = None
            self.instrument = None
            self.connected = False
            logging.info("Successfully disconnected from flow controller")
        except Exception as e:
            logging.error(f"Error during disconnect: {str(e)}")
            self.connected = False
            
    def set_flow_rate(self, flow_rate):
        """Set flow rate in ml/min using Propar protocol
        
        Args:
            flow_rate (float): The desired flow rate in ml/min
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.simulation_mode:
            print(f"Simulated: Setting flow rate to {flow_rate} ml/min")
            return True
        
        if not self.connected:
            logging.error("Not connected to device")
            return False
            
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                # Write setpoint (parameter 9)
                # Convert flow_rate to 0-32000 range
                setpoint = int((float(flow_rate) / 5000.0) * 32000)  # Assuming 5000 ml/min max
                self.instrument.writeParameter(9, setpoint)
                return True
            except Exception as e:
                logging.error(f"Set flow rate attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    try:
                        # Try to reconnect
                        self.disconnect()
                        if self.connect():
                            logging.info("Successfully reconnected")
                            continue
                    except Exception as reconnect_error:
                        logging.error(f"Reconnection failed: {str(reconnect_error)}")
        
        return False
    

    def calibrate_atmospheric_pressure(self, samples=10, settle_s=2.0, outlet_node=3):
        """
        Measure atmospheric baseline from the OUTLET sensor at no-flow.
        Stores the average as self.p_atm (bar[a]).
        Returns True on success, False otherwise.
        """
        if self.simulation_mode:
            # Simulate a plausible baseline
            self.p_atm = 1.013
            logging.info(f"[SIM] Calibrated p_atm = {self.p_atm:.4f} bar(a)")
            return True

        if not self.connected:
            logging.error("Not connected to device")
            return False

        try:
            # Small settle time at zero-flow before sampling
            time.sleep(settle_s)

            vals = []
            for _ in range(samples):
                try:
                    raw = self.instrument.readParameter(8, outlet_node)  # Parameter 8 from outlet node
                    if raw is not None:
                        # Convert raw (0..32000) to pressure (0..17 bar)
                        p = (float(raw) / 32000.0) * 17.0
                        vals.append(p)
                except Exception as e:
                    logging.error(f"Atmosphere sample read failed: {e}")
                time.sleep(0.2)

            if len(vals) >= max(2, samples // 2):
                self.p_atm = float(np.mean(vals))
                logging.info(f"Calibrated p_atm = {self.p_atm:.4f} bar(a) from {len(vals)} samples")
                return True
            else:
                logging.error("Not enough samples to calibrate p_atm")
                return False

        except Exception as e:
            logging.error(f"Calibration error: {e}")
            return False

        
    def get_measurements(self):
        """Get current measurements using Propar protocol
        
        Returns:
            dict: Dictionary containing flow_rate, inlet_pressure, outlet_pressure,
                  temperature, pressure_drop, and timestamp. None if error occurs.
        """
        if self.simulation_mode:
            # Simulate realistic values for development
            flow = np.random.normal(1000, 50)  # ml/min
            p1 = np.random.normal(2.5, 0.1)    # bar(a) - inlet pressure
            p2 = np.random.normal(1.5, 0.1)    # bar(a) - outlet pressure
            temp = np.random.normal(23, 1)     # °C
            
            return {
                'flow_rate': max(0, flow),
                'inlet_pressure': max(0, p1),
                'outlet_pressure': p2,
                'temperature': temp,
                'pressure_drop': max(0, p2 - 1.013),  # Difference from atmospheric
                'timestamp': time.time()
            }
            
        if not self.connected:
            logging.error("Not connected to device")
            return None
            
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                # Read flow measurement (parameter 8 = measure)
                measured = self.instrument.readParameter(8)
                logging.info(f"Raw flow reading (parameter 8): {measured}")
                
                if measured is not None:
                    # ProPar protocol: 32000 = 100%, max is 41943 (131.07%)
                    # Convert to percentage first
                    percentage = (float(measured) / 32000.0) * 100.0
                    # Then convert percentage to flow rate (0-5000 ml/min range)
                    flow_rate = (percentage / 100.0) * 5000
                    logging.info(f"Raw value {measured} = {percentage:.1f}% = {flow_rate:.1f} ml/min")
                else:
                    flow_rate = 0
                    logging.warning("Could not read flow measurement")
                    
                # Read pressure values using node-specific parameters
                # Node 1 = Flow controller
                # Node 2 = Pressure sensor 1 (inlet)
                # Node 3 = Pressure sensor 2 (outlet)
                # Parameter 8 = Measured value
                
                # Read inlet pressure (Node 2, Parameter 8)
                try:
                    inlet_p = self.instrument.readParameter(8, 2)  # Parameter 8 from node 2
                    logging.info(f"Raw inlet pressure reading from node 2: {inlet_p}")
                    
                    if inlet_p is not None:
                        # ProPar protocol: 32000 = 100%, max range is 0-17 bar
                        # Convert raw value directly to pressure
                        inlet_pressure = (float(inlet_p) / 32000.0) * 17.0
                        logging.info(f"Raw value {inlet_p} = {inlet_pressure:.3f} bar")
                    else:
                        inlet_pressure = 0
                        logging.warning("Could not read inlet pressure")
                except Exception as e:
                    logging.error(f"Error reading inlet pressure: {str(e)}")
                    inlet_pressure = 0
                
                # Read outlet pressure (Node 3, Parameter 8)
                try:
                    outlet_p = self.instrument.readParameter(8, 3)  # Parameter 8 from node 3
                    logging.info(f"Raw outlet pressure reading from node 3: {outlet_p}")
                    
                    if outlet_p is not None:
                        # ProPar protocol: 32000 = 100%, max range is 0-17 bar
                        # Convert raw value directly to pressure
                        outlet_pressure = (float(outlet_p) / 32000.0) * 17.0
                        logging.info(f"Raw value {outlet_p} = {outlet_pressure:.3f} bar")
                    else:
                        outlet_pressure = 1.013  # Default to atmospheric
                        logging.warning("Could not read outlet pressure, using atmospheric pressure")
                except Exception as e:
                    logging.error(f"Error reading outlet pressure: {str(e)}")
                    outlet_pressure = 1.013  # Default to atmospheric

                measurements = {
                    'flow': flow_rate,
                    'inlet': inlet_pressure,
                    'outlet': outlet_pressure,
                }
                
                # Read temperature
                temp = self.instrument.readParameter(142)
                logging.info(f"Raw temperature reading: {temp}")
                if temp is not None:
                    temperature = float(temp) / 32000.0 * 100  # Convert to °C
                    measurements['temp'] = temperature
                    logging.info(f"Converted temperature: {temperature:.1f} °C")
                else:
                    measurements['temp'] = 20  # Default temperature
                    logging.warning("Could not read temperature")
                
                # Use atmospheric pressure if no outlet pressure reading
                if 'outlet' not in measurements:
                    measurements['outlet'] = 1.013  # bar(a)
                    logging.warning("No outlet pressure reading found, using atmospheric pressure")
                    
                # Create return dictionary with all measurements
                inlet_p = float(measurements.get('inlet', 0))
                outlet_p = float(measurements.get('outlet', 1.013))
                baseline_atm = self.p_atm if self.p_atm is not None else 1.013

                return {
                    'flow_rate': float(measurements.get('flow', 0)),
                    'inlet_pressure': inlet_p,
                    'outlet_pressure': outlet_p,
                    'temperature': float(measurements.get('temp', 20)),
                    'pressure_drop': max(0, (outlet_p - baseline_atm)),  # outlet relative to calibrated atmosphere
                    'p_atm': baseline_atm,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logging.error(f"Get measurements attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    try:
                        # Try to reconnect
                        self.disconnect()
                        if self.connect():
                            logging.info("Successfully reconnected")
                            continue
                    except Exception as reconnect_error:
                        logging.error(f"Reconnection failed: {str(reconnect_error)}")
        
        return None

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
        self.latest_data = None
        self.reading_thread = None
        self.reading_active = False
        self.update_scheduled = False

        
        ###--------------------------------------------------------------------------------------###
        # Background reading thread           THESE ARE NOT IN NEW CODE, commented out for now
        self.reading_thread = None
        self.reading_active = False
        self.update_scheduled = False  # Prevent multiple update_readings schedules
        ###--------------------------------------------------------------------------------------###
        
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
        self.port_var = tk.StringVar(value="COM3")
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
        ttk.Button(data_frame, text="Calibrate Atmosphere", command=self._calibrate_atmosphere).grid(row=6, column=0, columnspan=3, pady=(10,0))

        
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
        ###--------------------------------------------------------------------------------------###
        self.update_scheduled = True   #not in other code 
        ###--------------------------------------------------------------------------------------###
        self.update_readings()
    
    ###------------------------------------------------------------------------------------------------------###
    def start_background_reading(self):
        if self.reading_active:
            return
        self.reading_active = True
        self.update_scheduled = True
        self.reading_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reading_thread.start()
        # Kick off the first UI update tick
        self.root.after(0, self.update_readings)

    def stop_background_reading(self):
        self.reading_active = False
        self.update_scheduled = False
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=1)
        self.reading_thread = None

    def _reader_loop(self):
        while self.reading_active and self.controller.connected:
            try:
                data = self.controller.get_measurements()   # <-- uses your existing method
                if data:
                    self.latest_data = data                 # <-- stash latest
            except Exception as e:
                logging.error(f"Reader error: {e}")
            time.sleep(0.5)  # small polling delay

    ###------------------------------------------------------------------------------------------------------###
    def toggle_simulation(self):
        """Toggle between simulation and real hardware mode (single-file Propar controller)"""
        new_mode = self.sim_mode_var.get()
        self.controller.simulation_mode = new_mode

        # If already connected, restart connection (and background reading) to apply mode
        if self.controller.connected:
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
            return

        if self.controller.connected and self.latest_data:
            data = self.latest_data
            try:
                self.flow_label.config(text=f"Flow: {data['flow_rate']:.1f} ml/min")
                self.p1_label.config(text=f"Inlet P: {data['inlet_pressure']:.3f} bar")
                self.p2_label.config(text=f"Outlet P: {data['outlet_pressure']:.3f} bar")
                self.dp_label.config(text=f"ΔP: {data['pressure_drop']*1000:.1f} mbar")
                self.temp_label.config(text=f"Temp: {data['temperature']:.1f} °C")
            except tk.TclError:
                self.update_scheduled = False
                return

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

    def _calibrate_atmosphere(self):
        if not self.controller.connected:
            messagebox.showerror("Error", "Controller not connected")
            return

        # Ensure flow is zero
        try:
            self.controller.set_flow_rate(0)
            time.sleep(2.0)  # allow settle
        except Exception as e:
            logging.error(f"Error setting flow to 0: {e}")
            return

        if self.controller.calibrate_atmospheric_pressure(samples=10, settle_s=2.0):
            messagebox.showinfo("Calibration", f"Atmospheric pressure calibrated: {self.controller.p_atm:.3f} bar(a)")
        else:
            messagebox.showerror("Calibration", "Failed to calibrate atmospheric pressure")

    
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

        # Sanitize sample name for Windows filenames
        sample = (self.current_sample or "run").strip()
        bad = '\\/:*?"<>|'
        for ch in bad:
            sample = sample.replace(ch, "_")

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"{sample}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                # ^^^ note: on some Tk builds, use 'initialfile' (not 'initialname')
            )
        except Exception as e:
            logging.error(f"Save dialog failed: {e}")
            messagebox.showerror("Error", f"Failed to open Save dialog:\n{e}")
            return

        if not filename:
            return  # user cancelled

        try:
            # Ensure data is DataFrame-friendly (convert any numpy types cleanly)
            df = pd.DataFrame(self.current_data)
            df.to_csv(filename, index=False)

            # Also save reference data if available
            if getattr(self, "reference_data", None):
                ref_filename = filename.replace(".csv", "_reference.csv")
                ref_df = pd.DataFrame(self.reference_data)
                ref_df.to_csv(ref_filename, index=False)

            messagebox.showinfo("Success", f"Data saved to:\n{filename}")

        except Exception as e:
            logging.exception("Failed to save data")
            messagebox.showerror("Error", f"Failed to save data:\n{e}")

    
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