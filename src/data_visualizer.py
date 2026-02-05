# Cell 1: Initialize Environment and Logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

# PyMC imports
import pymc as pm
import arviz as az

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('task_2_modeling.log')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f'Timestamp: {datetime.now()}')
logger.info(f'PyMC Version: {pm.__version__}')

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

class EDAVisualizer:
    """
    Create exploratory visualizations of the time series.
    """
    
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def plot_price_and_returns(self):
        """
        Plot price series and returns side by side.
        """
        self.logger.info('Creating price and returns visualizations')
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        # Price series
        axes[0].plot(self.data['Date'], self.data['Price'], color='darkblue', linewidth=1)
        axes[0].set_title('Brent Crude Oil Prices (1987-2022)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price (USD/barrel)')
        axes[0].grid(True, alpha=0.3)
        
        # Daily returns
        axes[1].plot(self.data['Date'], self.data['Returns'], color='darkgreen', linewidth=0.5)
        axes[1].set_title('Daily Percentage Returns', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Daily Return (%)')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        # Log returns distribution
        axes[2].hist(self.data['Log_Returns'], bins=100, color='darkred', alpha=0.7, edgecolor='black')
        axes[2].set_title('Distribution of Log Returns', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Log Return')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figures/task2_01_eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('EDA visualizations saved')
    
    def plot_rolling_statistics(self):
        """
        Plot rolling mean and volatility.
        """
        self.logger.info('Creating rolling statistics visualizations')
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        
        # Rolling mean
        rolling_mean_30 = self.data['Price'].rolling(window=30).mean()
        rolling_mean_90 = self.data['Price'].rolling(window=90).mean()
        
        axes[0].plot(self.data['Date'], self.data['Price'], label='Daily Price', linewidth=0.5, alpha=0.5)
        axes[0].plot(self.data['Date'], rolling_mean_30, label='30-Day Moving Avg', linewidth=2)
        axes[0].plot(self.data['Date'], rolling_mean_90, label='90-Day Moving Avg', linewidth=2)
        axes[0].set_title('Price with Moving Averages', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price (USD/barrel)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol_30 = self.data['Log_Returns'].rolling(window=30).std()
        rolling_vol_60 = self.data['Log_Returns'].rolling(window=60).std()
        
        axes[1].plot(self.data['Date'], rolling_vol_30, label='30-Day Rolling Vol', linewidth=1.5)
        axes[1].plot(self.data['Date'], rolling_vol_60, label='60-Day Rolling Vol', linewidth=1.5)
        axes[1].fill_between(self.data['Date'], rolling_vol_30, alpha=0.3)
        axes[1].set_title('Rolling Volatility (30-day and 60-day)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_xlabel('Year')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figures/task2_02_rolling_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Rolling statistics plot saved')

class PosteriorVisualizer:
    """
    Create comprehensive visualizations of posterior distributions.
    """
    
    def __init__(self, trace, data, cp_results, param_results):
        self.trace = trace
        self.data = data
        self.cp_results = cp_results
        self.param_results = param_results
        self.logger = logging.getLogger(__name__)
    
    def plot_change_point_posterior(self):
        """
        Plot posterior distribution of change point.
        """
        self.logger.info('Plotting change point posterior')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        tau_samples = self.cp_results['tau_samples']
        
        # Histogram of tau samples
        axes[0].hist(tau_samples, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(self.cp_results['tau_median'], color='red', linestyle='--', linewidth=2, label='Median')
        axes[0].axvline(self.cp_results['tau_hdi'][0], color='orange', linestyle=':', linewidth=2, label='95% HDI')
        axes[0].axvline(self.cp_results['tau_hdi'][1], color='orange', linestyle=':', linewidth=2)
        axes[0].set_title('Posterior Distribution of Change Point', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time Index (Days from Start)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Price series with change point overlay
        axes[1].plot(self.data['Date'], self.data['Price'], linewidth=1, alpha=0.6, label='Observed Price')
        cp_date = self.cp_results['cp_date']
        axes[1].axvline(cp_date, color='red', linestyle='--', linewidth=2, label=f'Change Point ({cp_date.date()})')
        axes[1].fill_betweenx(
            [self.data['Price'].min(), self.data['Price'].max()],
            self.data['Date'].iloc[int(self.cp_results['tau_hdi'][0])],
            self.data['Date'].iloc[int(self.cp_results['tau_hdi'][1])],
            alpha=0.2, color='orange', label='95% Credible Interval'
        )
        axes[1].set_title('Change Point in Time Series Context', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Price (USD/barrel)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figures/task2_04_change_point_posterior.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Change point posterior plot saved')
    
    def plot_parameter_posteriors(self):
        """
        Plot posterior distributions of parameters.
        """
        self.logger.info('Plotting parameter posteriors')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # mu1
        axes[0, 0].hist(self.param_results['mu1_samples'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.param_results['mu1_mean'], color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Posterior: Mean Price BEFORE Change Point', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('Price (USD/barrel)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # mu2
        axes[0, 1].hist(self.param_results['mu2_samples'], bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.param_results['mu2_mean'], color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Posterior: Mean Price AFTER Change Point', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('Price (USD/barrel)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sigma
        axes[1, 0].hist(self.param_results['sigma_samples'], bins=50, color='darkred', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.param_results['sigma_mean'], color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Posterior: Volatility (Sigma)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Volatility (USD/barrel)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparison of means
        axes[1, 1].hist(self.param_results['mu1_samples'], bins=40, alpha=0.5, label='Before', color='steelblue')
        axes[1, 1].hist(self.param_results['mu2_samples'], bins=40, alpha=0.5, label='After', color='darkgreen')
        axes[1, 1].set_title('Mean Price Comparison', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Price (USD/barrel)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figures/task2_05_parameter_posteriors.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Parameter posterior plots saved')