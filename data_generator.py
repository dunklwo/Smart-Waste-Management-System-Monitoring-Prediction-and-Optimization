class EnhancedDataGenerator:
    """Generate realistic waste data with weather patterns"""
    
    def __init__(self, num_bins=50):
        self.num_bins = num_bins
        self.waste_types = ['Plastic', 'Organic', 'Glass', 'Metal', 'Paper', 'E-waste']
        self.zones = ['Residential-A', 'Residential-B', 'Commercial', 'Industrial', 'Market']
        self.weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Sunny']
        
    def generate_bins(self):
        """Generate bin configurations"""
        bins = []
        for i in range(1, self.num_bins + 1):
            bin_data = {
                'bin_id': f'BIN{str(i).zfill(3)}',
                'zone': np.random.choice(self.zones),
                'location': f'Street {i}, Sector {(i % 10) + 1}',
                'latitude': round(28.6 + np.random.uniform(-0.05, 0.05), 6),
                'longitude': round(77.2 + np.random.uniform(-0.05, 0.05), 6),
                'capacity': np.random.choice([100, 150, 200, 250]),
                'waste_type': np.random.choice(self.waste_types),
                'installation_date': datetime.now() - timedelta(days=np.random.randint(0, 365))
            }
            bins.append(bin_data)
        return pd.DataFrame(bins)
    
    def generate_weather_data(self, days=30):
        """Generate weather data affecting waste generation"""
        weather_data = []
        for day in range(days):
            date = datetime.now() - timedelta(days=days-day)
            weather = {
                'date': date,
                'temperature': round(np.random.normal(25, 8), 1),  # Mean 25°C, std 8
                'humidity': round(np.random.normal(60, 15), 1),
                'condition': np.random.choice(self.weather_conditions, 
                                            p=[0.4, 0.3, 0.2, 0.1]),
                'is_holiday': np.random.random() < 0.1,  # 10% holidays
                'is_weekend': date.weekday() >= 5
            }
            weather_data.append(weather)
        return pd.DataFrame(weather_data)
    
    def generate_historical_data(self, bins_df, weather_df, days=30):
        """Generate realistic historical waste data with correlations"""
        historical_data = []
        
        for _, bin_info in bins_df.iterrows():
            current_level = np.random.uniform(0, 30)
            
            for _, weather in weather_df.iterrows():
                for hour in range(24):
                    timestamp = weather['date'] + timedelta(hours=hour)
                    
                    # Base fill rate
                    base_rate = np.random.normal(1.5, 0.5)
                    
                    # Waste type factor
                    if bin_info['waste_type'] == 'Organic':
                        base_rate *= 1.8
                    elif bin_info['waste_type'] == 'Plastic':
                        base_rate *= 1.5
                    
                    # Zone factor
                    if bin_info['zone'] in ['Commercial', 'Market']:
                        base_rate *= 1.6
                    elif bin_info['zone'] == 'Industrial':
                        base_rate *= 1.3
                    
                    # Time of day factor
                    if 6 <= hour <= 10 or 17 <= hour <= 21:  # Peak hours
                        base_rate *= 1.8
                    elif 0 <= hour <= 6:  # Night hours
                        base_rate *= 0.3
                    
                    # Weather factor
                    if weather['condition'] == 'Rainy':
                        base_rate *= 0.7  # Less waste in rain
                    elif weather['condition'] == 'Sunny':
                        base_rate *= 1.2
                    
                    # Weekend/Holiday factor
                    if weather['is_weekend'] or weather['is_holiday']:
                        if bin_info['zone'] in ['Residential-A', 'Residential-B']:
                            base_rate *= 1.4
                        else:
                            base_rate *= 0.6
                    
                    # Update level
                    current_level = min(current_level + base_rate, bin_info['capacity'])
                    
                    record = {
                        'timestamp': timestamp,
                        'bin_id': bin_info['bin_id'],
                        'zone': bin_info['zone'],
                        'location': bin_info['location'],
                        'waste_type': bin_info['waste_type'],
                        'capacity': bin_info['capacity'],
                        'fill_level': round(current_level, 2),
                        'fill_percentage': round((current_level / bin_info['capacity']) * 100, 2),
                        'temperature': weather['temperature'],
                        'humidity': weather['humidity'],
                        'weather_condition': weather['condition']
                    }
                    historical_data.append(record)
                    
                    # Reset if collected (90%+ full)
                    if current_level >= bin_info['capacity'] * 0.9:
                        current_level = np.random.uniform(5, 15)
        
        return pd.DataFrame(historical_data)