import sqlite3
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class WasteDatabase:
    """SQLite Database for persistent storage"""
    
    def __init__(self, db_name='waste_management.db'):
        self.db_name = db_name
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables"""
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Bins table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bins (
                bin_id TEXT PRIMARY KEY,
                location TEXT,
                latitude REAL,
                longitude REAL,
                capacity INTEGER,
                waste_type TEXT,
                zone TEXT,
                installation_date DATE
            )
        ''')
        
        # Waste readings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS waste_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bin_id TEXT,
                timestamp DATETIME,
                fill_level REAL,
                fill_percentage REAL,
                temperature REAL,
                humidity REAL,
                weather_condition TEXT,
                FOREIGN KEY (bin_id) REFERENCES bins(bin_id)
            )
        ''')
        
        # Complaints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citizen_name TEXT,
                contact TEXT,
                bin_id TEXT,
                location TEXT,
                complaint_type TEXT,
                description TEXT,
                severity TEXT,
                status TEXT DEFAULT 'Open',
                timestamp DATETIME,
                image_path TEXT,
                resolution_notes TEXT
            )
        ''')
        
        # Collection routes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_date DATE,
                truck_id TEXT,
                route_json TEXT,
                total_distance REAL,
                bins_collected INTEGER,
                status TEXT
            )
        ''')
        
        self.conn.commit()
        print("✓ Database initialized successfully")
    
    def insert_bin(self, bin_data):
        """Insert bin data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO bins 
            (bin_id, location, latitude, longitude, capacity, waste_type, zone, installation_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (bin_data['bin_id'], bin_data['location'], bin_data['latitude'],
              bin_data['longitude'], bin_data['capacity'], bin_data['waste_type'],
              bin_data['zone'], bin_data.get('installation_date', datetime.now())))
        self.conn.commit()
    
    def insert_reading(self, reading_data):
        """Insert waste reading"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO waste_readings 
            (bin_id, timestamp, fill_level, fill_percentage, temperature, humidity, weather_condition)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (reading_data['bin_id'], reading_data['timestamp'], reading_data['fill_level'],
              reading_data['fill_percentage'], reading_data.get('temperature'),
              reading_data.get('humidity'), reading_data.get('weather_condition')))
        self.conn.commit()
    
    def insert_complaint(self, complaint_data):
        """Insert citizen complaint"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO complaints 
            (citizen_name, contact, bin_id, location, complaint_type, description, 
             severity, timestamp, image_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (complaint_data['citizen_name'], complaint_data['contact'],
              complaint_data.get('bin_id'), complaint_data['location'],
              complaint_data['complaint_type'], complaint_data['description'],
              complaint_data['severity'], datetime.now(), 
              complaint_data.get('image_path'), 'Open'))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_bins(self):
        """Get all bins"""
        return pd.read_sql_query("SELECT * FROM bins", self.conn)
    
    def get_readings(self, bin_id=None, days=7):
        """Get readings for specific bin or all bins"""
        date_from = datetime.now() - timedelta(days=days)
        if bin_id:
            query = f"SELECT * FROM waste_readings WHERE bin_id='{bin_id}' AND timestamp >= '{date_from}'"
        else:
            query = f"SELECT * FROM waste_readings WHERE timestamp >= '{date_from}'"
        return pd.read_sql_query(query, self.conn)
    
    def get_complaints(self, status='all'):
        """Get complaints"""
        if status == 'all':
            query = "SELECT * FROM complaints ORDER BY timestamp DESC"
        else:
            query = f"SELECT * FROM complaints WHERE status='{status}' ORDER BY timestamp DESC"
        return pd.read_sql_query(query, self.conn)
    
    def update_complaint_status(self, complaint_id, status, resolution_notes=''):
        """Update complaint status"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE complaints 
            SET status=?, resolution_notes=? 
            WHERE id=?
        ''', (status, resolution_notes, complaint_id))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()