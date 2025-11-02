"""
Forecasting module for demand prediction using Random Forest regression.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from typing import Dict, Tuple


def forecast_demand_random_forest(
    historical_demand: pd.DataFrame,
    forecast_horizon: int = 12,
    n_lags: int = 6,
    n_estimators: int = 100,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Forecast demand using Random Forest regression with lag features.
    
    As per thesis methodology: uses lag features (e.g., 6 months) to predict future demand.
    
    Args:
        historical_demand: DataFrame with columns ['period', 'customer_id', 'demand']
        forecast_horizon: Number of periods to forecast
        n_lags: Number of lag features to use (default: 6)
        n_estimators: Number of trees in Random Forest
        random_state: Random seed
    
    Returns:
        Tuple of (forecast DataFrame, accuracy metrics dictionary)
    """
    forecasts = []
    all_metrics = {}
    
    for customer_id in historical_demand['customer_id'].unique():
        customer_data = historical_demand[
            historical_demand['customer_id'] == customer_id
        ].sort_values('period').copy()
        
        # Create lag features
        for lag in range(1, n_lags + 1):
            customer_data[f'Lag{lag}'] = customer_data['demand'].shift(lag)
        # Add seasonal/time features
        customer_data['month'] = customer_data['period'] % 12
        customer_data['sin_season'] = np.sin(2 * np.pi * customer_data['period'] / 12)
        customer_data['cos_season'] = np.cos(2 * np.pi * customer_data['period'] / 12)
        # Rolling statistics
        for w in [3, 6, 12]:
            customer_data[f'roll_mean_{w}'] = customer_data['demand'].rolling(w).mean()
            customer_data[f'roll_std_{w}'] = customer_data['demand'].rolling(w).std()
        
        customer_data = customer_data.dropna()
        
        if len(customer_data) < n_lags + forecast_horizon:
            # Not enough data - use simple trend
            X_train = customer_data[['period']].values if len(customer_data) > 1 else [[1]]
            y_train = customer_data['demand'].values if len(customer_data) > 1 else [customer_data['demand'].mean()]
            model = LinearRegression()
            model.fit(X_train, y_train)
            max_period = customer_data['period'].max()
            for period in range(max_period + 1, max_period + forecast_horizon + 1):
                pred = model.predict([[period]])[0] if len(customer_data) > 1 else customer_data['demand'].mean()
                forecasts.append({
                    'period': period,
                    'customer_id': customer_id,
                    'demand': max(0, pred),
                    'method': 'linear_fallback'
                })
            continue
        
        # Train-test split (use last forecast_horizon as test)
        split_idx = len(customer_data) - forecast_horizon if len(customer_data) > forecast_horizon else len(customer_data) - 1
        train = customer_data.iloc[:split_idx]
        test = customer_data.iloc[split_idx:] if split_idx < len(customer_data) else None
        
        # Prepare features
        lag_cols = [f'Lag{i}' for i in range(1, n_lags + 1)]
        extra_cols = ['month', 'sin_season', 'cos_season', 'roll_mean_3', 'roll_std_3', 'roll_mean_6', 'roll_std_6', 'roll_mean_12', 'roll_std_12']
        feature_cols = lag_cols + extra_cols
        X_train = train[feature_cols].values
        y_train = train['demand'].values
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=2
        )
        model.fit(X_train, y_train)
        
        # Forecast future periods using iterative prediction
        max_period = customer_data['period'].max()
        last_values = customer_data['demand'].tail(n_lags).values
        # Prepare last known extra features
        last_period = int(customer_data['period'].iloc[-1])
        last_roll = {
            'roll_mean_3': float(customer_data['roll_mean_3'].iloc[-1]),
            'roll_std_3': float(customer_data['roll_std_3'].iloc[-1]),
            'roll_mean_6': float(customer_data['roll_mean_6'].iloc[-1]),
            'roll_std_6': float(customer_data['roll_std_6'].iloc[-1]),
            'roll_mean_12': float(customer_data['roll_mean_12'].iloc[-1]),
            'roll_std_12': float(customer_data['roll_std_12'].iloc[-1])
        }
        
        for period in range(max_period + 1, max_period + forecast_horizon + 1):
            # Update seasonal features
            month = period % 12
            sin_season = np.sin(2 * np.pi * period / 12)
            cos_season = np.cos(2 * np.pi * period / 12)
            # Rolling updates (approximate: update rolling mean/std with last predictions)
            series = last_values[-max(12, n_lags):]
            roll_mean_3 = float(np.mean(series[-3:])) if len(series) >= 3 else float(np.mean(series))
            roll_std_3 = float(np.std(series[-3:])) if len(series) >= 3 else float(np.std(series))
            roll_mean_6 = float(np.mean(series[-6:])) if len(series) >= 6 else float(np.mean(series))
            roll_std_6 = float(np.std(series[-6:])) if len(series) >= 6 else float(np.std(series))
            roll_mean_12 = float(np.mean(series[-12:])) if len(series) >= 12 else float(np.mean(series))
            roll_std_12 = float(np.std(series[-12:])) if len(series) >= 12 else float(np.std(series))

            # Build feature vector
            X_pred = np.concatenate([
                last_values[-n_lags:],
                np.array([month, sin_season, cos_season, roll_mean_3, roll_std_3, roll_mean_6, roll_std_6, roll_mean_12, roll_std_12])
            ]).reshape(1, -1)
            pred = model.predict(X_pred)[0]
            pred = max(0, pred)  # Ensure non-negative
            
            forecasts.append({
                'period': period,
                'customer_id': customer_id,
                'demand': pred,
                'method': 'random_forest'
            })
            
            # Update last_values for next iteration
            last_values = np.append(last_values, pred)
        
        # Calculate accuracy on test set if available
        if test is not None and len(test) > 0:
            X_test = test[feature_cols].values
            y_test = test['demand'].values
            y_pred_test = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            all_metrics[customer_id] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    forecast_df = pd.DataFrame(forecasts)
    
    # Aggregate metrics across all customers
    if all_metrics:
        avg_metrics = {
            'MAE': np.mean([m['MAE'] for m in all_metrics.values()]),
            'RMSE': np.mean([m['RMSE'] for m in all_metrics.values()]),
            'MAPE': np.mean([m['MAPE'] for m in all_metrics.values()])
        }
    else:
        avg_metrics = {'MAE': 0, 'RMSE': 0, 'MAPE': 0}
    
    return forecast_df, avg_metrics


def forecast_demand(
    historical_demand: pd.DataFrame,
    forecast_horizon: int = 12,
    method: str = 'random_forest'
) -> pd.DataFrame:
    """
    Forecast future demand using different methods.
    
    Args:
        historical_demand: DataFrame with columns ['period', 'customer_id', 'demand']
        forecast_horizon: Number of periods to forecast
        method: Forecasting method ('random_forest', 'moving_average', 'linear_trend', 'seasonal')
    
    Returns:
        DataFrame with forecasted demand
    """
    if method == 'random_forest':
        forecast_df, _ = forecast_demand_random_forest(
            historical_demand,
            forecast_horizon=forecast_horizon
        )
        return forecast_df
    
    # Fallback to other methods
    max_period = historical_demand['period'].max()
    forecasts = []
    
    for customer_id in historical_demand['customer_id'].unique():
        customer_data = historical_demand[
            historical_demand['customer_id'] == customer_id
        ].sort_values('period')
        
        if method == 'moving_average':
            # Simple moving average
            window = min(4, len(customer_data))
            avg_demand = customer_data['demand'].tail(window).mean()
            
            for period in range(max_period + 1, max_period + forecast_horizon + 1):
                forecasts.append({
                    'period': period,
                    'customer_id': customer_id,
                    'demand': avg_demand,
                    'method': method
                })
        
        elif method == 'linear_trend':
            # Linear regression trend
            X = customer_data[['period']].values
            y = customer_data['demand'].values
            
            if len(X) > 1:
                model = LinearRegression()
                model.fit(X, y)
                
                for period in range(max_period + 1, max_period + forecast_horizon + 1):
                    forecast = model.predict([[period]])[0]
                    forecasts.append({
                        'period': period,
                        'customer_id': customer_id,
                        'demand': max(0, forecast),
                        'method': method
                    })
            else:
                # Fallback to mean
                avg_demand = customer_data['demand'].mean()
                for period in range(max_period + 1, max_period + forecast_horizon + 1):
                    forecasts.append({
                        'period': period,
                        'customer_id': customer_id,
                        'demand': avg_demand,
                        'method': method
                    })
        
        elif method == 'seasonal':
            # Seasonal naive with trend
            if len(customer_data) >= 12:
                seasonal_pattern = customer_data['demand'].tail(12).values
                trend = (customer_data['demand'].iloc[-1] - customer_data['demand'].iloc[-12]) / 12
                
                for i, period in enumerate(range(max_period + 1, max_period + forecast_horizon + 1)):
                    seasonal_idx = i % 12
                    forecast = seasonal_pattern[seasonal_idx] + trend * (i // 12 + 1)
                    forecasts.append({
                        'period': period,
                        'customer_id': customer_id,
                        'demand': max(0, forecast),
                        'method': method
                    })
            else:
                avg_demand = customer_data['demand'].mean()
                for period in range(max_period + 1, max_period + forecast_horizon + 1):
                    forecasts.append({
                        'period': period,
                        'customer_id': customer_id,
                        'demand': avg_demand,
                        'method': method
                    })
    
    return pd.DataFrame(forecasts)


def calculate_forecast_accuracy(
    actual: pd.DataFrame,
    forecast: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        actual: Actual demand DataFrame
        forecast: Forecasted demand DataFrame
    
    Returns:
        Dictionary with accuracy metrics (MAE, RMSE, MAPE)
    """
    merged = pd.merge(
        actual, forecast,
        on=['period', 'customer_id'],
        suffixes=('_actual', '_forecast')
    )
    
    mae = (merged['demand_actual'] - merged['demand_forecast']).abs().mean()
    rmse = np.sqrt(((merged['demand_actual'] - merged['demand_forecast']) ** 2).mean())
    
    # Avoid division by zero in MAPE
    non_zero_mask = merged['demand_actual'] > 0
    if non_zero_mask.sum() > 0:
        mape = (
            (merged.loc[non_zero_mask, 'demand_actual'] - 
             merged.loc[non_zero_mask, 'demand_forecast']).abs() /
            merged.loc[non_zero_mask, 'demand_actual']
        ).mean() * 100
    else:
        mape = 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

