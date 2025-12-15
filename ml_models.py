from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

class MLModelComparison:
    """Train and compare multiple ML models"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def prepare_features(self, data):
        """Prepare features for ML models"""
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        df = df.sort_values(['bin_id', 'timestamp'])
        df['fill_level_lag1'] = df.groupby('bin_id')['fill_level'].shift(1)
        df['fill_level_lag24'] = df.groupby('bin_id')['fill_level'].shift(24)
        
        # Rolling statistics
        df['fill_level_rolling_mean'] = df.groupby('bin_id')['fill_level'].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        
        # One-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=['waste_type', 'zone', 'weather_condition'], drop_first=True)
        
        # Drop rows with NaN
        df = df.dropna()
        
        return df
    
    def train_and_evaluate(self, data, target_col='fill_level'):
        """Train all models and compare performance"""
        print("\n🤖 Training Multiple ML Models...")
        print("="*80)
        
        # Prepare data
        df = self.prepare_features(data)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in 
                       ['bin_id', 'timestamp', 'location', 'fill_level', 'fill_percentage']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Number of features: {len(feature_cols)}\n")
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                           scoring='r2')
                
                self.results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'accuracy': test_r2 * 100,  # Percentage
                    'y_test': y_test,
                    'y_pred': y_pred_test
                }
                
                print(f"  ✓ Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
                print(f"  ✓ Test RMSE: {test_rmse:.2f}")
                print(f"  ✓ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}\n")
                continue
        
        # Find best model
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print("="*80)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Accuracy: {self.results[self.best_model_name]['accuracy']:.2f}%")
        print("="*80 + "\n")
        
        return self.results
    
    def predict_future(self, bin_data, hours_ahead=48):
        """Predict future fill levels"""
        if self.best_model is None:
            return None
        
        # This is a simplified version - in production, you'd need to prepare features properly
        predictions = []
        current_time = datetime.now()
        
        for hour in range(hours_ahead):
            pred_time = current_time + timedelta(hours=hour)
            # Simplified prediction - in reality, prepare full feature set
            predictions.append({
                'timestamp': pred_time,
                'predicted_fill_percentage': min(100, bin_data['fill_percentage'] + hour * 2)
            })
        
        return pd.DataFrame(predictions)