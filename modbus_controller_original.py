import logging
from pymodbus.client.serial import ModbusSerialClient
import time
import pymodbus

# Enable debug logging for pymodbus
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

# Print pymodbus version at startup
print(f"Using pymodbus version: {pymodbus.__version__}")

class BronkhorstModbusController:
    def __init__(self, port, address=1, baudrate=9600, parity='N'):
        self.port = port
        self.address = address
        self.baudrate = baudrate
        self.parity = parity

        # Create a Modbus RTU client to communicate with the device
        # Configure for pymodbus 3.9.2 with optimal settings for Bronkhorst devices
        self.client = ModbusSerialClient(
            port=self.port,
            baudrate=self.baudrate,
            parity=self.parity,
            stopbits=1,    # Standard Modbus RTU setting
            bytesize=8,    # Standard Modbus RTU setting
            timeout=2.0,   # Increased timeout for more reliable communication
            retries=6      # Number of retry attempts
        )

    def connect(self):
        print(f"Connecting to {self.port} at {self.baudrate} baud, parity {self.parity}, address {self.address}")
        # Clear any existing connection first
        try:
            self.client.close()
        except:
            pass
            
        # Try to connect with proper delay
        time.sleep(0.5)  # Short delay before connecting
        connected = self.client.connect()
        print(f"Connect result: {connected}")
        
        if connected:
            time.sleep(1.5)  # Longer delay after successful connection for device initialization
        
        return connected

    def disconnect(self):
        print("Disconnecting...")
        self.client.close()

    def set_flow_rate(self, flow_rate):
        try:
            value = int((flow_rate / 5000) * 32000)
            print(f"Writing value {value} to register 34 (set flow rate)")
            result = self.client.write_register(34, value, slave=self.address)
            print(f"Write result: {result}")
            if result.isError():
                print(f"Modbus error: {result}")
            return not result.isError()
        except Exception as e:
            print(f"Error al fijar flujo: {e}")
            return False

    def get_all_measurements(self):
        try:
            def read_register(register):
                print(f"Reading register {register}")
                result = self.client.read_input_registers(register, count=1, slave=self.address)
                print(f"Read result: {result}")
                if not result.isError():
                    return result.registers[0]
                else:
                    print(f"Modbus error reading register {register}: {result}")
                    raise Exception(f"Error leyendo registro {register}")

            flow_raw = read_register(33)
            p1_raw = read_register(64)
            p2_raw = read_register(65)
            temp_raw = read_register(66)

            return {
                'flow_rate': (flow_raw / 32000) * 5000,
                'inlet_pressure': p1_raw / 1000,
                'outlet_pressure': p2_raw / 1000,
                'temperature': temp_raw / 10,
                'pressure_drop': max(0, (p1_raw - p2_raw) / 1000),
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"Error al leer datos: {e}")
            return None
