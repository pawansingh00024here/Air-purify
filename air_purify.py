import requests
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import threading
import logging
from collections import defaultdict, deque
import statistics
import warnings
import hashlib
import csv
import os
import sys
from abc import ABC, abstractmethod
import re
from urllib.parse import urlencode
import socket
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('air_quality.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AQICategory(Enum):
    """Air Quality Index categories based on EPA standards"""
    GOOD = (0, 50, "Good", "green", "Air quality is satisfactory, and air pollution poses little or no risk")
    MODERATE = (51, 100, "Moderate", "yellow", "Air quality is acceptable; however, there may be a risk for some people")
    UNHEALTHY_SENSITIVE = (101, 150, "Unhealthy for Sensitive Groups", "orange", 
                          "Members of sensitive groups may experience health effects")
    UNHEALTHY = (151, 200, "Unhealthy", "red", "Everyone may begin to experience health effects")
    VERY_UNHEALTHY = (201, 300, "Very Unhealthy", "purple", "Health alert: everyone may experience serious health effects")
    HAZARDOUS = (301, 500, "Hazardous", "maroon", "Health warnings of emergency conditions")

    def __init__(self, min_val: int, max_val: int, label: str, color: str, description: str):
        self.min_val = min_val
        self.max_val = max_val
        self.label = label
        self.color = color
        self.description = description

    @classmethod
    def get_category(cls, aqi_value: float) -> 'AQICategory':
        """Get AQI category for a given AQI value"""
        for category in cls:
            if category.min_val <= aqi_value <= category.max_val:
                return category
        return cls.HAZARDOUS

    @classmethod
    def get_health_recommendations(cls, category: 'AQICategory') -> List[str]:
        """Get health recommendations for a given AQI category"""
        recommendations = {
            cls.GOOD: [
                "Air quality is ideal for outdoor activities",
                "No health precautions needed",
                "Enjoy your usual outdoor activities"
            ],
            cls.MODERATE: [
                "Unusually sensitive people should consider reducing prolonged outdoor exertion",
                "Generally acceptable air quality",
                "Monitor symptoms if you're sensitive to air pollution"
            ],
            cls.UNHEALTHY_SENSITIVE: [
                "Sensitive groups should reduce prolonged outdoor exertion",
                "People with respiratory or heart conditions should limit outdoor activities",
                "Children and older adults should take it easy"
            ],
            cls.UNHEALTHY: [
                "Everyone should reduce prolonged outdoor exertion",
                "Sensitive groups should avoid outdoor activities",
                "Keep windows closed if possible",
                "Use air purifiers indoors"
            ],
            cls.VERY_UNHEALTHY: [
                "Everyone should avoid prolonged outdoor exertion",
                "Sensitive groups should remain indoors",
                "Keep all windows closed",
                "Use air purifiers and masks if going outside"
            ],
            cls.HAZARDOUS: [
                "Everyone should avoid all outdoor activities",
                "Remain indoors and keep windows closed",
                "Run air purifiers on high",
                "Wear N95 masks if you must go outside",
                "Seek medical attention if experiencing symptoms"
            ]
        }
        return recommendations.get(category, [])


class Pollutant(Enum):
    """Types of air pollutants monitored"""
    PM25 = ("PM2.5", "Particulate Matter 2.5", "μg/m³", 12.0, 35.4)
    PM10 = ("PM10", "Particulate Matter 10", "μg/m³", 54, 154)
    O3 = ("O3", "Ozone", "ppb", 0, 54)
    NO2 = ("NO2", "Nitrogen Dioxide", "ppb", 0, 53)
    SO2 = ("SO2", "Sulfur Dioxide", "ppb", 0, 35)
    CO = ("CO", "Carbon Monoxide", "ppm", 0, 4.4)

    def __init__(self, code: str, name: str, unit: str, good_max: float, moderate_max: float):
        self.code = code
        self.full_name = name
        self.unit = unit
        self.good_max = good_max
        self.moderate_max = moderate_max


@dataclass
class Location:
    """Represents a geographical location for air quality monitoring"""
    name: str
    latitude: float
    longitude: float
    city: str
    country: str
    timezone: str = "UTC"
    elevation: Optional[float] = None
    population: Optional[int] = None
    
    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate distance to another location in kilometers using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = np.radians(self.latitude), np.radians(self.longitude)
        lat2, lon2 = np.radians(other.latitude), np.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert location to dictionary"""
        return asdict(self)


@dataclass
class AirQualityReading:
    """Represents a single air quality reading"""
    timestamp: datetime
    location: Location
    aqi: float
    pollutants: Dict[str, float]
    dominant_pollutant: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    pressure: Optional[float] = None
    source: str = "unknown"
    
    @property
    def category(self) -> AQICategory:
        """Get the AQI category for this reading"""
        return AQICategory.get_category(self.aqi)
    
    @property
    def is_healthy(self) -> bool:
        """Check if air quality is healthy"""
        return self.aqi <= 100
    
    @property
    def needs_alert(self) -> bool:
        """Check if this reading requires an alert"""
        return self.aqi > 150
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['category'] = self.category.label
        return data


class DataSource(ABC):
    """Abstract base class for air quality data sources"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def fetch_current(self, location: Location) -> Optional[AirQualityReading]:
        """Fetch current air quality data for a location"""
        pass
    
    @abstractmethod
    def fetch_historical(self, location: Location, start_date: datetime, 
                        end_date: datetime) -> List[AirQualityReading]:
        """Fetch historical air quality data"""
        pass
    
    @abstractmethod
    def fetch_forecast(self, location: Location, days: int = 7) -> List[AirQualityReading]:
        """Fetch air quality forecast"""
        pass


class OpenAQDataSource(DataSource):
    """Data source implementation for OpenAQ API"""
    
    BASE_URL = "https://api.openaq.org/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("OpenAQ", api_key)
    
    def fetch_current(self, location: Location) -> Optional[AirQualityReading]:
        """Fetch current air quality from OpenAQ"""
        try:
            params = {
                'coordinates': f"{location.latitude},{location.longitude}",
                'radius': 25000,  # 25km radius
                'limit': 1,
                'order_by': 'lastUpdated'
            }
            
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            
            response = requests.get(
                f"{self.BASE_URL}/latest",
                params=params,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('results'):
                result = data['results'][0]
                
                pollutants = {}
                for measurement in result.get('measurements', []):
                    param = measurement.get('parameter')
                    value = measurement.get('value')
                    if param and value is not None:
                        pollutants[param] = value
                
                # Calculate AQI (simplified)
                aqi = self._calculate_aqi(pollutants)
                dominant = self._get_dominant_pollutant(pollutants)
                
                return AirQualityReading(
                    timestamp=datetime.fromisoformat(result['date']['utc'].replace('Z', '+00:00')),
                    location=location,
                    aqi=aqi,
                    pollutants=pollutants,
                    dominant_pollutant=dominant,
                    source=self.name
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching current data: {e}")
            return None
    
    def fetch_historical(self, location: Location, start_date: datetime, 
                        end_date: datetime) -> List[AirQualityReading]:
        """Fetch historical data from OpenAQ"""
        readings = []
        try:
            params = {
                'coordinates': f"{location.latitude},{location.longitude}",
                'radius': 25000,
                'date_from': start_date.isoformat(),
                'date_to': end_date.isoformat(),
                'limit': 1000
            }
            
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            
            response = requests.get(
                f"{self.BASE_URL}/measurements",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Process results (simplified for brevity)
            # In a real implementation, this would properly aggregate measurements
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
        
        return readings
    
    def fetch_forecast(self, location: Location, days: int = 7) -> List[AirQualityReading]:
        """OpenAQ doesn't provide forecasts, so return empty list"""
        return []
    
    def _calculate_aqi(self, pollutants: Dict[str, float]) -> float:
        """Calculate AQI from pollutant concentrations (simplified EPA formula)"""
        if not pollutants:
            return 0
        
        aqi_values = []
        
        # PM2.5 breakpoints
        if 'pm25' in pollutants:
            pm25 = pollutants['pm25']
            breakpoints = [
                (0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 500.4, 301, 500)
            ]
            aqi_values.append(self._calculate_sub_aqi(pm25, breakpoints))
        
        # PM10 breakpoints
        if 'pm10' in pollutants:
            pm10 = pollutants['pm10']
            breakpoints = [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500)
            ]
            aqi_values.append(self._calculate_sub_aqi(pm10, breakpoints))
        
        # O3 breakpoints (8-hour)
        if 'o3' in pollutants:
            o3 = pollutants['o3']
            breakpoints = [
                (0, 54, 0, 50),
                (55, 70, 51, 100),
                (71, 85, 101, 150),
                (86, 105, 151, 200),
                (106, 200, 201, 300)
            ]
            aqi_values.append(self._calculate_sub_aqi(o3, breakpoints))
        
        return max(aqi_values) if aqi_values else 0
    
    def _calculate_sub_aqi(self, concentration: float, breakpoints: List[Tuple]) -> float:
        """Calculate sub-index for a pollutant"""
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
        return 500  # Hazardous if beyond all breakpoints
    
    def _get_dominant_pollutant(self, pollutants: Dict[str, float]) -> str:
        """Determine the dominant pollutant"""
        if not pollutants:
            return "unknown"
        
        max_aqi = 0
        dominant = "unknown"
        
        for pollutant, value in pollutants.items():
            if pollutant == 'pm25':
                breakpoints = [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
                             (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)]
            elif pollutant == 'pm10':
                breakpoints = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
                             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)]
            elif pollutant == 'o3':
                breakpoints = [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
                             (86, 105, 151, 200), (106, 200, 201, 300)]
            else:
                continue
            
            aqi = self._calculate_sub_aqi(value, breakpoints)
            if aqi > max_aqi:
                max_aqi = aqi
                dominant = pollutant
        
        return dominant


class SimulatedDataSource(DataSource):
    """Simulated data source for testing and demonstration"""
    
    def __init__(self):
        super().__init__("Simulated")
    
    def fetch_current(self, location: Location) -> Optional[AirQualityReading]:
        """Generate simulated current air quality data"""
        base_aqi = np.random.normal(75, 25)
        base_aqi = max(0, min(500, base_aqi))
        
        pollutants = {
            'pm25': np.random.normal(25, 10),
            'pm10': np.random.normal(45, 15),
            'o3': np.random.normal(40, 15),
            'no2': np.random.normal(30, 10),
            'so2': np.random.normal(5, 3),
            'co': np.random.normal(0.5, 0.2)
        }
        
        # Ensure non-negative values
        pollutants = {k: max(0, v) for k, v in pollutants.items()}
        
        return AirQualityReading(
            timestamp=datetime.now(),
            location=location,
            aqi=base_aqi,
            pollutants=pollutants,
            dominant_pollutant='pm25',
            temperature=np.random.normal(20, 10),
            humidity=np.random.normal(60, 20),
            wind_speed=np.random.normal(10, 5),
            wind_direction=np.random.uniform(0, 360),
            pressure=np.random.normal(1013, 10),
            source=self.name
        )
    
    def fetch_historical(self, location: Location, start_date: datetime, 
                        end_date: datetime) -> List[AirQualityReading]:
        """Generate simulated historical data"""
        readings = []
        current = start_date
        
        while current <= end_date:
            base_aqi = np.random.normal(75, 25)
            base_aqi = max(0, min(500, base_aqi))
            
            pollutants = {
                'pm25': np.random.normal(25, 10),
                'pm10': np.random.normal(45, 15),
                'o3': np.random.normal(40, 15),
                'no2': np.random.normal(30, 10),
                'so2': np.random.normal(5, 3),
                'co': np.random.normal(0.5, 0.2)
            }
            
            pollutants = {k: max(0, v) for k, v in pollutants.items()}
            
            reading = AirQualityReading(
                timestamp=current,
                location=location,
                aqi=base_aqi,
                pollutants=pollutants,
                dominant_pollutant='pm25',
                temperature=np.random.normal(20, 10),
                humidity=np.random.normal(60, 20),
                wind_speed=np.random.normal(10, 5),
                wind_direction=np.random.uniform(0, 360),
                pressure=np.random.normal(1013, 10),
                source=self.name
            )
            readings.append(reading)
            
            current += timedelta(hours=1)
        
        return readings
    
    def fetch_forecast(self, location: Location, days: int = 7) -> List[AirQualityReading]:
        """Generate simulated forecast data"""
        readings = []
        current = datetime.now()
        
        for i in range(days * 24):
            forecast_time = current + timedelta(hours=i)
            
            # Trend: slightly improving over time
            base_aqi = np.random.normal(75 - i*0.1, 25)
            base_aqi = max(0, min(500, base_aqi))
            
            pollutants = {
                'pm25': np.random.normal(25 - i*0.05, 10),
                'pm10': np.random.normal(45 - i*0.08, 15),
                'o3': np.random.normal(40, 15),
                'no2': np.random.normal(30, 10),
                'so2': np.random.normal(5, 3),
                'co': np.random.normal(0.5, 0.2)
            }
            
            pollutants = {k: max(0, v) for k, v in pollutants.items()}
            
            reading = AirQualityReading(
                timestamp=forecast_time,
                location=location,
                aqi=base_aqi,
                pollutants=pollutants,
                dominant_pollutant='pm25',
                temperature=np.random.normal(20, 10),
                humidity=np.random.normal(60, 20),
                wind_speed=np.random.normal(10, 5),
                wind_direction=np.random.uniform(0, 360),
                pressure=np.random.normal(1013, 10),
                source=f"{self.name}_forecast"
            )
            readings.append(reading)
        
        return readings


class DatabaseManager:
    """Manages SQLite database operations for air quality data"""
    
    def __init__(self, db_path: str = "air_quality.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Locations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                city TEXT,
                country TEXT,
                timezone TEXT,
                elevation REAL,
                population INTEGER,
                UNIQUE(latitude, longitude)
            )
        ''')
        
        # Readings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                location_id INTEGER NOT NULL,
                aqi REAL NOT NULL,
                dominant_pollutant TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction REAL,
                pressure REAL,
                source TEXT,
                FOREIGN KEY (location_id) REFERENCES locations(id)
            )
        ''')
        
        # Pollutants table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pollutants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reading_id INTEGER NOT NULL,
                pollutant_type TEXT NOT NULL,
                value REAL NOT NULL,
                FOREIGN KEY (reading_id) REFERENCES readings(id)
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                location_id INTEGER NOT NULL,
                aqi_level REAL NOT NULL,
                category TEXT NOT NULL,
                message TEXT,
                acknowledged BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (location_id) REFERENCES locations(id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_readings_location ON readings(location_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pollutants_reading ON pollutants(reading_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_location ON alerts(location_id)')
        
        self.conn.commit()
    
    def save_location(self, location: Location) -> int:
        """Save a location and return its ID"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO locations (name, latitude, longitude, city, country, timezone, elevation, population)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (location.name, location.latitude, location.longitude, location.city, 
              location.country, location.timezone, location.elevation, location.population))
        
        cursor.execute('''
            SELECT id FROM locations WHERE latitude = ? AND longitude = ?
        ''', (location.latitude, location.longitude))
        
        location_id = cursor.fetchone()[0]
        self.conn.commit()
        
        return location_id
    
    def save_reading(self, reading: AirQualityReading) -> int:
        """Save an air quality reading and return its ID"""
        location_id = self.save_location(reading.location)
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO readings (timestamp, location_id, aqi, dominant_pollutant, 
                                temperature, humidity, wind_speed, wind_direction, pressure, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (reading.timestamp.isoformat(), location_id, reading.aqi, reading.dominant_pollutant,
              reading.temperature, reading.humidity, reading.wind_speed, 
              reading.wind_direction, reading.pressure, reading.source))
        
        reading_id = cursor.lastrowid
        
        # Save pollutants
        for pollutant_type, value in reading.pollutants.items():
            cursor.execute('''
                INSERT INTO pollutants (reading_id, pollutant_type, value)
                VALUES (?, ?, ?)
            ''', (reading_id, pollutant_type, value))
        
        self.conn.commit()
        
        return reading_id
    
    def get_readings(self, location_id: Optional[int] = None, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    limit: Optional[int] = None) -> List[Dict]:
        """Retrieve readings from database"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM readings WHERE 1=1"
        params = []
        
        if location_id:
            query += " AND location_id = ?"
            params.append(location_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        readings = []
        
        for row in cursor.fetchall():
            reading_dict = dict(zip(columns, row))
            readings.append(reading_dict)
        
        return readings
    
    def save_alert(self, location: Location, aqi_level: float, category: str, message: str):
        """Save an alert to the database"""
        location_id = self.save_location(location)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (timestamp, location_id, aqi_level, category, message)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), location_id, aqi_level, category, message))
        
        self.conn.commit()
    
    def get_statistics(self, location_id: Optional[int] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Calculate statistics for readings"""
        readings = self.get_readings(location_id, start_date, end_date)
        
        if not readings:
            return {}
        
        aqi_values = [r['aqi'] for r in readings]
        
        return {
            'count': len(readings),
            'mean_aqi': statistics.mean(aqi_values),
            'median_aqi': statistics.median(aqi_values),
            'min_aqi': min(aqi_values),
            'max_aqi': max(aqi_values),
            'stdev_aqi': statistics.stdev(aqi_values) if len(aqi_values) > 1 else 0,
            'good_days': sum(1 for aqi in aqi_values if aqi <= 50),
            'moderate_days': sum(1 for aqi in aqi_values if 51 <= aqi <= 100),
            'unhealthy_sensitive_days': sum(1 for aqi in aqi_values if 101 <= aqi <= 150),
            'unhealthy_days': sum(1 for aqi in aqi_values if 151 <= aqi <= 200),
            'very_unhealthy_days': sum(1 for aqi in aqi_values if 201 <= aqi <= 300),
            'hazardous_days': sum(1 for aqi in aqi_values if aqi > 300)
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class AlertSystem:
    """System for managing air quality alerts"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.subscribers = defaultdict(list)  # email -> locations
        self.thresholds = {
            'moderate': 100,
            'unhealthy_sensitive': 150,
            'unhealthy': 200,
            'very_unhealthy': 300
        }
    
    def subscribe(self, email: str, location: Location):
        """Subscribe to alerts for a location"""
        self.subscribers[email].append(location)
        logger.info(f"Subscribed {email} to alerts for {location.name}")
    
    def unsubscribe(self, email: str, location: Optional[Location] = None):
        """Unsubscribe from alerts"""
        if location:
            if email in self.subscribers:
                self.subscribers[email] = [loc for loc in self.subscribers[email] 
                                          if loc.name != location.name]
        else:
            if email in self.subscribers:
                del self.subscribers[email]
    
    def check_reading(self, reading: AirQualityReading) -> Optional[str]:
        """Check if a reading triggers an alert"""
        if reading.aqi > self.thresholds['unhealthy_sensitive']:
            category = reading.category
            message = f"Air Quality Alert for {reading.location.name}: {category.label} (AQI: {reading.aqi:.0f})"
            
            self.db_manager.save_alert(
                reading.location,
                reading.aqi,
                category.label,
                message
            )
            
            return message
        
        return None
    
    def send_alert(self, email: str, message: str):
        """Send alert to subscriber (simulated)"""
        logger.info(f"ALERT sent to {email}: {message}")
        # In a real implementation, this would send an actual email
    
    def process_reading(self, reading: AirQualityReading):
        """Process a reading and send alerts if necessary"""
        alert_message = self.check_reading(reading)
        
        if alert_message:
            for email, locations in self.subscribers.items():
                for location in locations:
                    if location.name == reading.location.name:
                        self.send_alert(email, alert_message)


class AirQualityPredictor:
    """Machine learning-based air quality predictor"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.models = {}  # location_id -> model
    
    def prepare_features(self, readings: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector from readings"""
        if not readings:
            return np.array([]), np.array([])
        
        features = []
        targets = []
        
        for reading in readings:
            # Extract time features
            timestamp = datetime.fromisoformat(reading['timestamp'])
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            day_of_year = timestamp.timetuple().tm_yday
            
            # Cyclical encoding for time features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            feature_vector = [
                hour_sin, hour_cos,
                day_sin, day_cos,
                reading.get('temperature', 0) or 0,
                reading.get('humidity', 0) or 0,
                reading.get('wind_speed', 0) or 0,
                reading.get('pressure', 0) or 0,
            ]
            
            features.append(feature_vector)
            targets.append(reading['aqi'])
        
        return np.array(features), np.array(targets)
    
    def train_simple_model(self, location_id: int, lookback_days: int = 30):
        """Train a simple linear regression model"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if len(readings) < 10:
            logger.warning(f"Not enough data to train model for location {location_id}")
            return
        
        X, y = self.prepare_features(readings)
        
        if len(X) == 0:
            return
        
        # Simple linear regression using numpy
        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # Normal equation: w = (X^T X)^-1 X^T y
        try:
            weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            self.models[location_id] = weights
            logger.info(f"Trained model for location {location_id}")
        except np.linalg.LinAlgError:
            logger.error(f"Failed to train model for location {location_id}")
    
    def predict(self, location_id: int, timestamp: datetime, 
                temperature: float, humidity: float, 
                wind_speed: float, pressure: float) -> Optional[float]:
        """Predict AQI for given conditions"""
        if location_id not in self.models:
            logger.warning(f"No model available for location {location_id}")
            return None
        
        weights = self.models[location_id]
        
        # Prepare features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        features = np.array([1, hour_sin, hour_cos, day_sin, day_cos, 
                            temperature, humidity, wind_speed, pressure])
        
        prediction = np.dot(features, weights)
        
        return max(0, min(500, prediction))  # Clamp to valid AQI range


class Analyzer:
    """Air quality data analyzer"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def analyze_trends(self, location_id: int, days: int = 30) -> Dict[str, Any]:
        """Analyze trends over a specified period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if not readings:
            return {'error': 'No data available'}
        
        aqi_values = [r['aqi'] for r in readings]
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in readings]
        
        # Calculate trend using linear regression
        x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        y = np.array(aqi_values)
        
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            trend_direction = 'improving' if slope < 0 else 'worsening' if slope > 0 else 'stable'
        else:
            slope = 0
            trend_direction = 'insufficient data'
        
        # Identify peak pollution times
        hourly_aqi = defaultdict(list)
        for reading in readings:
            hour = datetime.fromisoformat(reading['timestamp']).hour
            hourly_aqi[hour].append(reading['aqi'])
        
        avg_hourly_aqi = {hour: statistics.mean(values) 
                         for hour, values in hourly_aqi.items()}
        
        peak_hour = max(avg_hourly_aqi.items(), key=lambda x: x[1]) if avg_hourly_aqi else (0, 0)
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': slope,
            'peak_pollution_hour': peak_hour[0],
            'peak_pollution_aqi': peak_hour[1],
            'average_aqi': statistics.mean(aqi_values),
            'worst_day': max(readings, key=lambda x: x['aqi']),
            'best_day': min(readings, key=lambda x: x['aqi'])
        }
    
    def compare_locations(self, location_ids: List[int], days: int = 30) -> Dict[str, Any]:
        """Compare air quality across multiple locations"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        location_stats = {}
        
        for location_id in location_ids:
            stats = self.db_manager.get_statistics(location_id, start_date, end_date)
            location_stats[location_id] = stats
        
        return location_stats
    
    def identify_correlations(self, location_id: int, days: int = 30) -> Dict[str, float]:
        """Identify correlations between weather and air quality"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if len(readings) < 10:
            return {'error': 'Insufficient data'}
        
        aqi = np.array([r['aqi'] for r in readings])
        temperature = np.array([r.get('temperature') or 0 for r in readings])
        humidity = np.array([r.get('humidity') or 0 for r in readings])
        wind_speed = np.array([r.get('wind_speed') or 0 for r in readings])
        
        correlations = {}
        
        if np.std(temperature) > 0:
            correlations['temperature'] = np.corrcoef(aqi, temperature)[0, 1]
        
        if np.std(humidity) > 0:
            correlations['humidity'] = np.corrcoef(aqi, humidity)[0, 1]
        
        if np.std(wind_speed) > 0:
            correlations['wind_speed'] = np.corrcoef(aqi, wind_speed)[0, 1]
        
        return correlations


class Visualizer:
    """Air quality data visualization"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        sns.set_style("whitegrid")
    
    def plot_time_series(self, location_id: int, days: int = 30, save_path: Optional[str] = None):
        """Plot AQI time series"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if not readings:
            logger.warning("No data to plot")
            return
        
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in readings]
        aqi_values = [r['aqi'] for r in readings]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, aqi_values, linewidth=2)
        
        # Add AQI category bands
        plt.axhspan(0, 50, alpha=0.1, color='green', label='Good')
        plt.axhspan(51, 100, alpha=0.1, color='yellow', label='Moderate')
        plt.axhspan(101, 150, alpha=0.1, color='orange', label='Unhealthy (Sensitive)')
        plt.axhspan(151, 200, alpha=0.1, color='red', label='Unhealthy')
        plt.axhspan(201, 300, alpha=0.1, color='purple', label='Very Unhealthy')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.title(f'Air Quality Index - Last {days} Days', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_distribution(self, location_id: int, days: int = 30, save_path: Optional[str] = None):
        """Plot AQI distribution"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if not readings:
            logger.warning("No data to plot")
            return
        
        aqi_values = [r['aqi'] for r in readings]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(aqi_values, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('AQI', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('AQI Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        categories = [AQICategory.get_category(aqi).label for aqi in aqi_values]
        category_counts = pd.Series(categories).value_counts()
        
        colors = [AQICategory.get_category(aqi).color for aqi in category_counts.index]
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                startangle=90)
        plt.title('AQI Category Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_heatmap(self, location_id: int, days: int = 30, save_path: Optional[str] = None):
        """Plot hourly heatmap of AQI"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if not readings:
            logger.warning("No data to plot")
            return
        
        # Create hour x day matrix
        data = defaultdict(lambda: defaultdict(list))
        
        for reading in readings:
            timestamp = datetime.fromisoformat(reading['timestamp'])
            hour = timestamp.hour
            day = timestamp.strftime('%Y-%m-%d')
            data[day][hour].append(reading['aqi'])
        
        # Calculate averages
        days_list = sorted(data.keys())
        hours = list(range(24))
        
        matrix = np.zeros((len(days_list), 24))
        
        for i, day in enumerate(days_list):
            for hour in hours:
                if hour in data[day]:
                    matrix[i, hour] = statistics.mean(data[day][hour])
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(matrix, cmap='RdYlGn_r', cbar_kws={'label': 'AQI'},
                   xticklabels=hours, yticklabels=[d[-5:] for d in days_list])
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Date', fontsize=12)
        plt.title('Air Quality Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class ReportGenerator:
    """Generate comprehensive air quality reports"""
    
    def __init__(self, db_manager: DatabaseManager, analyzer: Analyzer):
        self.db_manager = db_manager
        self.analyzer = analyzer
    
    def generate_summary_report(self, location_id: int, days: int = 30) -> str:
        """Generate a text summary report"""
        stats = self.db_manager.get_statistics(location_id, None, None)
        trends = self.analyzer.analyze_trends(location_id, days)
        
        report = []
        report.append("=" * 70)
        report.append("AIR QUALITY SUMMARY REPORT")
        report.append("=" * 70)
        report.append(f"\nReport Period: Last {days} days")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "-" * 70)
        report.append("\nSTATISTICS:")
        report.append(f"  Total Readings: {stats.get('count', 0)}")
        report.append(f"  Average AQI: {stats.get('mean_aqi', 0):.1f}")
        report.append(f"  Median AQI: {stats.get('median_aqi', 0):.1f}")
        report.append(f"  Min AQI: {stats.get('min_aqi', 0):.1f}")
        report.append(f"  Max AQI: {stats.get('max_aqi', 0):.1f}")
        report.append(f"  Std Dev: {stats.get('stdev_aqi', 0):.1f}")
        
        report.append("\n" + "-" * 70)
        report.append("\nAQI CATEGORY BREAKDOWN:")
        report.append(f"  Good Days (0-50): {stats.get('good_days', 0)}")
        report.append(f"  Moderate Days (51-100): {stats.get('moderate_days', 0)}")
        report.append(f"  Unhealthy for Sensitive (101-150): {stats.get('unhealthy_sensitive_days', 0)}")
        report.append(f"  Unhealthy (151-200): {stats.get('unhealthy_days', 0)}")
        report.append(f"  Very Unhealthy (201-300): {stats.get('very_unhealthy_days', 0)}")
        report.append(f"  Hazardous (300+): {stats.get('hazardous_days', 0)}")
        
        report.append("\n" + "-" * 70)
        report.append("\nTRENDS:")
        report.append(f"  Trend Direction: {trends.get('trend_direction', 'N/A')}")
        report.append(f"  Peak Pollution Hour: {trends.get('peak_pollution_hour', 0)}:00")
        report.append(f"  Peak Hour AQI: {trends.get('peak_pollution_aqi', 0):.1f}")
        
        if 'worst_day' in trends:
            worst = trends['worst_day']
            report.append(f"\n  Worst Day: {worst['timestamp']}")
            report.append(f"    AQI: {worst['aqi']:.1f}")
        
        if 'best_day' in trends:
            best = trends['best_day']
            report.append(f"\n  Best Day: {best['timestamp']}")
            report.append(f"    AQI: {best['aqi']:.1f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def export_to_csv(self, location_id: int, output_path: str, days: int = 30):
        """Export data to CSV file"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        readings = self.db_manager.get_readings(location_id, start_date, end_date)
        
        if not readings:
            logger.warning("No data to export")
            return
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'aqi', 'dominant_pollutant', 'temperature', 
                         'humidity', 'wind_speed', 'pressure', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for reading in readings:
                writer.writerow({
                    'timestamp': reading['timestamp'],
                    'aqi': reading['aqi'],
                    'dominant_pollutant': reading['dominant_pollutant'],
                    'temperature': reading.get('temperature'),
                    'humidity': reading.get('humidity'),
                    'wind_speed': reading.get('wind_speed'),
                    'pressure': reading.get('pressure'),
                    'source': reading.get('source')
                })
        
        logger.info(f"Data exported to {output_path}")


class AirQualityMonitor:
    """Main air quality monitoring system"""
    
    def __init__(self, db_path: str = "air_quality.db"):
        self.db_manager = DatabaseManager(db_path)
        self.data_sources = {}
        self.alert_system = AlertSystem(self.db_manager)
        self.predictor = AirQualityPredictor(self.db_manager)
        self.analyzer = Analyzer(self.db_manager)
        self.visualizer = Visualizer(self.db_manager)
        self.report_generator = ReportGenerator(self.db_manager, self.analyzer)
        self.monitored_locations = []
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def add_data_source(self, source: DataSource):
        """Add a data source to the system"""
        self.data_sources[source.name] = source
        logger.info(f"Added data source: {source.name}")
    
    def add_location(self, location: Location):
        """Add a location to monitor"""
        self.monitored_locations.append(location)
        self.db_manager.save_location(location)
        logger.info(f"Added monitoring location: {location.name}")
    
    def collect_current_data(self, location: Location) -> Optional[AirQualityReading]:
        """Collect current data from all available sources"""
        for source in self.data_sources.values():
            try:
                reading = source.fetch_current(location)
                if reading:
                    self.db_manager.save_reading(reading)
                    self.alert_system.process_reading(reading)
                    return reading
            except Exception as e:
                logger.error(f"Error fetching from {source.name}: {e}")
        
        return None
    
    def collect_historical_data(self, location: Location, days: int = 30):
        """Collect historical data for a location"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for source in self.data_sources.values():
            try:
                readings = source.fetch_historical(location, start_date, end_date)
                for reading in readings:
                    self.db_manager.save_reading(reading)
                logger.info(f"Collected {len(readings)} historical readings from {source.name}")
            except Exception as e:
                logger.error(f"Error fetching historical data from {source.name}: {e}")
    
    def start_monitoring(self, interval_minutes: int = 60):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                for location in self.monitored_locations:
                    try:
                        reading = self.collect_current_data(location)
                        if reading:
                            logger.info(f"Collected reading for {location.name}: AQI {reading.aqi:.1f}")
                    except Exception as e:
                        logger.error(f"Error monitoring {location.name}: {e}")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Started monitoring (interval: {interval_minutes} minutes)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring")
    
    def get_current_status(self, location: Location) -> Dict[str, Any]:
        """Get current air quality status for a location"""
        reading = self.collect_current_data(location)
        
        if not reading:
            return {'error': 'No data available'}
        
        category = reading.category
        
        return {
            'timestamp': reading.timestamp.isoformat(),
            'location': location.name,
            'aqi': reading.aqi,
            'category': category.label,
            'color': category.color,
            'description': category.description,
            'health_recommendations': AQICategory.get_health_recommendations(category),
            'dominant_pollutant': reading.dominant_pollutant,
            'pollutants': reading.pollutants,
            'temperature': reading.temperature,
            'humidity': reading.humidity,
            'wind_speed': reading.wind_speed
        }
    
    def generate_forecast(self, location: Location, days: int = 7) -> List[Dict[str, Any]]:
        """Generate air quality forecast"""
        forecasts = []
        
        for source in self.data_sources.values():
            try:
                forecast_readings = source.fetch_forecast(location, days)
                
                for reading in forecast_readings:
                    forecasts.append({
                        'timestamp': reading.timestamp.isoformat(),
                        'aqi': reading.aqi,
                        'category': reading.category.label,
                        'source': reading.source
                    })
                
                if forecast_readings:
                    break  # Use first successful forecast
                    
            except Exception as e:
                logger.error(f"Error generating forecast from {source.name}: {e}")
        
        return forecasts
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        self.db_manager.close()


def main():
    """Main demonstration function"""
    print("=" * 70)
    print("AIR QUALITY MONITORING AND ANALYSIS SYSTEM")
    print("=" * 70)
    print()
    
    # Initialize the system
    monitor = AirQualityMonitor("air_quality_demo.db")
    
    # Add simulated data source (for demonstration)
    monitor.add_data_source(SimulatedDataSource())
    
    # Add sample locations
    locations = [
        Location("New York", 40.7128, -74.0060, "New York", "USA", "America/New_York"),
        Location("Los Angeles", 34.0522, -118.2437, "Los Angeles", "USA", "America/Los_Angeles"),
        Location("London", 51.5074, -0.1278, "London", "UK", "Europe/London"),
        Location("Tokyo", 35.6762, 139.6503, "Tokyo", "Japan", "Asia/Tokyo"),
        Location("Mumbai", 19.0760, 72.8777, "Mumbai", "India", "Asia/Kolkata")
    ]
    
    for location in locations:
        monitor.add_location(location)
    
    # Collect some historical data
    print("Collecting historical data...")
    for location in locations[:2]:  # Just first two for demo
        monitor.collect_historical_data(location, days=7)
    
    # Get current status
    print("\n" + "=" * 70)
    print("CURRENT AIR QUALITY STATUS")
    print("=" * 70)
    
    for location in locations[:2]:
        status = monitor.get_current_status(location)
        if 'error' not in status:
            print(f"\n{status['location']}:")
            print(f"  AQI: {status['aqi']:.1f} ({status['category']})")
            print(f"  Description: {status['description']}")
            print(f"  Dominant Pollutant: {status['dominant_pollutant']}")
    
    # Generate report
    print("\n" + "=" * 70)
    location_id = 1  # First location
    report = monitor.report_generator.generate_summary_report(location_id, days=7)
    print(report)
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    try:
        monitor.visualizer.plot_time_series(location_id, days=7, save_path="aqi_timeseries.png")
        print("✓ Time series plot saved to: aqi_timeseries.png")
    except Exception as e:
        print(f"✗ Time series plot failed: {e}")
    
    try:
        monitor.visualizer.plot_distribution(location_id, days=7, save_path="aqi_distribution.png")
        print("✓ Distribution plot saved to: aqi_distribution.png")
    except Exception as e:
        print(f"✗ Distribution plot failed: {e}")
    
    # Export data
    print("\n" + "=" * 70)
    print("EXPORTING DATA")
    print("=" * 70)
    
    try:
        monitor.report_generator.export_to_csv(location_id, "air_quality_data.csv", days=7)
        print("✓ Data exported to: air_quality_data.csv")
    except Exception as e:
        print(f"✗ Data export failed: {e}")
    
    # Demonstrate forecasting
    print("\n" + "=" * 70)
    print("AIR QUALITY FORECAST")
    print("=" * 70)
    
    forecast = monitor.generate_forecast(locations[0], days=3)
    if forecast:
        print(f"\n7-Day Forecast for {locations[0].name}:")
        for i, f in enumerate(forecast[:24:6]):  # Show every 6 hours for first day
            print(f"  {f['timestamp']}: AQI {f['aqi']:.1f} ({f['category']})")
    
    # Demonstrate alert system
    print("\n" + "=" * 70)
    print("ALERT SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    monitor.alert_system.subscribe("user@example.com", locations[0])
    print("✓ Subscribed user@example.com to alerts for", locations[0].name)
    
    # Cleanup
    print("\n" + "=" * 70)
    print("SYSTEM CLEANUP")
    print("=" * 70)
    monitor.cleanup()
    print("✓ System cleaned up successfully")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - air_quality_demo.db (SQLite database)")
    print("  - aqi_timeseries.png (visualization)")
    print("  - aqi_distribution.png (visualization)")
    print("  - air_quality_data.csv (exported data)")
    print("  - air_quality.log (system log)")
    print()


if __name__ == "__main__":
    main()