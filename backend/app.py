# File: app.py - Main Flask Application

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING MODULE
# ============================================================================

class DataManager:
    """
    Modular class for managing data loading and caching.
    """
    def __init__(self):
        self.price_data = None
        self.events_data = None
        self.model_results = None
        self.load_all_data()
    
    def load_all_data(self):
        """Load all required data on initialization."""
        try:
            logger.info('Loading data...')
            
            # Load price data
            self.price_data = pd.read_csv('data/BrentOilPrices.csv')
            self.price_data['Date'] = pd.to_datetime(self.price_data['Date'],errors='coerce')
            self.price_data = self.price_data.sort_values('Date')
            
            # Load events data
            self.events_data = pd.read_csv('data/major_events.csv')
            self.events_data['date'] = pd.to_datetime(self.events_data['date'])
            
            # Load model results (from Task 2 pickle file)
            import pickle
            with open('data/model_results.pkl', 'rb') as f:
                self.model_results = pickle.load(f)
            
            logger.info('All data loaded successfully')
        except Exception as e:
            logger.error(f'Error loading data: {str(e)}')
            raise

data_manager = DataManager()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

@app.route('/api/v1/prices', methods=['GET'])
def get_prices():
    """Get historical price data with optional filtering."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        prices = data_manager.price_data.copy()
        
        if start_date:
            prices = prices[prices['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            prices = prices[prices['Date'] <= pd.to_datetime(end_date)]
        
        return jsonify({
            'data': prices.to_dict('records'),
            'count': len(prices),
            'date_range': {
                'start': prices['Date'].min().isoformat(),
                'end': prices['Date'].max().isoformat()
            }
        }), 200
    except Exception as e:
        logger.error(f'Error fetching prices: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/change-point', methods=['GET'])
def get_change_point():
    try:
        cp_date_str = data_manager.model_results['cp_date']
        
        # Convert string to datetime if it's not already
        if isinstance(cp_date_str, str):
            # Try common formats from your data (adjust if needed)
            try:
                cp_date = pd.to_datetime(cp_date_str, format='%d-%b-%y')  # e.g. 15-Sep-08
            except ValueError:
                try:
                    cp_date = pd.to_datetime(cp_date_str)  # auto-detect ISO/other
                except:
                    cp_date = pd.to_datetime(cp_date_str, errors='coerce')
                    if pd.isna(cp_date):
                        raise ValueError(f"Cannot parse date: {cp_date_str}")
        else:
            cp_date = cp_date_str  # already datetime

        results = {
            'detected_date': cp_date.isoformat(),  # now safe
            'mean_before': float(data_manager.model_results['mu1_mean']),
            'mean_after': float(data_manager.model_results['mu2_mean']),
            'absolute_change': float(data_manager.model_results['absolute_change']),
            'percent_change': float(data_manager.model_results['percent_change']),
            'credible_interval': {
                'lower': int(data_manager.model_results['hdi_lower']),
                'upper': int(data_manager.model_results['hdi_upper'])
            },
            'volatility': float(data_manager.model_results['sigma_mean'])
        }
        return jsonify(results), 200
    except Exception as e:
        logger.error(f'Error fetching change point: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())  # more detail in logs
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/events', methods=['GET'])
def get_events():
    """Get major events data."""
    try:
        return jsonify({
            'data': data_manager.events_data.to_dict('records'),
            'count': len(data_manager.events_data)
        }), 200
    except Exception as e:
        logger.error(f'Error fetching events: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/statistics', methods=['GET'])
def get_statistics():
    """Get summary statistics."""
    try:
        prices = data_manager.price_data['Price']
        stats = {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std_dev': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'range': float(prices.max() - prices.min()),
            'records': len(prices)
        }
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f'Error calculating statistics: {str(e)}')
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Internal server error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
