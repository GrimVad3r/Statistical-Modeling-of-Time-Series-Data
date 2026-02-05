# Cell 1: Import Libraries and Configure Logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('task_1_analysis.log')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f'Timestamp: {datetime.now()}')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
class DataLoader:
    """
    Modular class for loading and validating Brent oil price data.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)
        self.df = None
    
    def load_data(self):
        """
        Load CSV data and perform initial validation.
        """
        try:
            self.logger.info(f'Loading data from {self.filepath}')
            self.df = pd.read_csv(self.filepath)
            self.logger.info(f'Data loaded successfully. Shape: {self.df.shape}')
            return self.df
        except Exception as e:
            self.logger.error(f'Error loading data: {str(e)}')
            raise
    
    def preprocess_data(self):
        """
        Convert date column and handle missing values.
        """
        try:
            # Convert date column
            self.logger.info('Converting Date column to datetime format')
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            
            # Sort by date
            self.logger.info('Sorting data by date')
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            
            # Convert price to numeric
            self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
            
            # Check for missing values
            missing_count = self.df.isnull().sum().sum()
            if missing_count > 0:
                self.logger.warning(f'Found {missing_count} missing values')
                self.df = self.df.dropna()
                self.logger.info(f'After removing missing values: {self.df.shape}')
            else:
                self.logger.info('No missing values found')
            
            return self.df
        except Exception as e:
            self.logger.error(f'Error preprocessing data: {str(e)}')
            raise
    
    def validate_data(self):
        """
        Validate data quality and consistency.
        """
        self.logger.info('Validating data quality')
        
        validation_results = {
            'total_records': len(self.df),
            'date_range': f"{self.df['Date'].min().date()} to {self.df['Date'].max().date()}",
            'price_range': f"${self.df['Price'].min():.2f} - ${self.df['Price'].max():.2f}",
            'null_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        
        print('\nDATA VALIDATION RESULTS')
        print('=' * 50)
        for key, value in validation_results.items():
            print(f'{key}: {value}')
        
        self.logger.info('Data validation completed successfully')
        return validation_results