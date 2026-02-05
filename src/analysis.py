# Cell 1: Import Libraries and Configure Logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import sys
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss

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

class DataAnalysisWorkflow:
    """
    Modularized workflow for Brent oil price analysis.
    Follows data science best practices with structured steps.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workflow_steps = []
    
    def document_workflow(self):
        """
        Define and document the complete analysis pipeline.
        """
        self.workflow_steps = [
            {
                'step': 1,
                'phase': 'Data Loading & Preparation',
                'tasks': [
                    'Load Brent oil price CSV',
                    'Convert date column to datetime format',
                    'Handle missing values',
                    'Sort data chronologically',
                    'Validate data quality'
                ],
                'dependencies': None,
                'tools': ['pandas', 'numpy']
            },
            {
                'step': 2,
                'phase': 'Exploratory Data Analysis (EDA)',
                'tasks': [
                    'Analyze time series properties (trend, seasonality)',
                    'Test stationarity (ADF test)',
                    'Calculate log returns',
                    'Analyze volatility patterns',
                    'Visualize price movements and shocks'
                ],
                'dependencies': [1],
                'tools': ['statsmodels', 'matplotlib', 'seaborn']
            },
            {
                'step': 3,
                'phase': 'Research & Event Compilation',
                'tasks': [
                    'Research major geopolitical events',
                    'Research OPEC policy decisions',
                    'Research economic/sanctions events',
                    'Compile structured event dataset',
                    'Create event CSV with dates and descriptions'
                ],
                'dependencies': None,
                'tools': ['csv', 'pandas']
            },
            {
                'step': 4,
                'phase': 'Model Understanding',
                'tasks': [
                    'Study Bayesian change point theory',
                    'Review PyMC documentation',
                    'Understand switch points and priors',
                    'Plan model architecture',
                    'Document assumptions and limitations'
                ],
                'dependencies': [2],
                'tools': ['PyMC', 'documentation']
            },
            {
                'step': 5,
                'phase': 'Change Point Modeling',
                'tasks': [
                    'Build Bayesian change point model in PyMC',
                    'Define priors and likelihood',
                    'Run MCMC sampling',
                    'Check convergence diagnostics',
                    'Extract and interpret posterior distributions'
                ],
                'dependencies': [1, 4],
                'tools': ['PyMC', 'arviz']
            },
            {
                'step': 6,
                'phase': 'Event Association & Interpretation',
                'tasks': [
                    'Map detected change points to dates',
                    'Match change points with historical events',
                    'Quantify price impact per event',
                    'Calculate percentage changes',
                    'Formulate hypotheses about causation'
                ],
                'dependencies': [3, 5],
                'tools': ['pandas', 'numpy', 'visualization']
            },
            {
                'step': 7,
                'phase': 'Dashboard Development',
                'tasks': [
                    'Design Flask API endpoints',
                    'Create React frontend components',
                    'Build interactive visualizations',
                    'Implement event highlighting',
                    'Add date range filtering'
                ],
                'dependencies': [5, 6],
                'tools': ['Flask', 'React', 'Recharts']
            },
            {
                'step': 8,
                'phase': 'Reporting & Communication',
                'tasks': [
                    'Create comprehensive report',
                    'Generate summary statistics',
                    'Produce final visualizations',
                    'Write executive summary',
                    'Document limitations and future work'
                ],
                'dependencies': [5, 6, 7],
                'tools': ['pandas', 'matplotlib', 'documentation']
            }
        ]
        
        self.logger.info(f'Workflow documented with {len(self.workflow_steps)} major phases')
        return self.workflow_steps
    
    def display_workflow(self):
        """
        Display the workflow in a readable format.
        """
        print('\n' + '='*80)
        print('DATA ANALYSIS WORKFLOW - BRENT OIL PRICES')
        print('='*80 + '\n')
        
        for step in self.workflow_steps:
            print(f"\nStep {step['step']}: {step['phase'].upper()}")
            print('-' * 60)
            for task in step['tasks']:
                print(f"  • {task}")
            print(f"  Tools: {', '.join(step['tools'])}")
            if step['dependencies']:
                print(f"  Dependencies: Steps {step['dependencies']}")
        print('\n' + '='*80)

class TimeSeriesAnalyzer:
    """
    Analyze key properties of the Brent oil price time series.
    """
    
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def calculate_log_returns(self):
        """
        Calculate log returns for stationarity analysis.
        """
        self.logger.info('Calculating log returns')
        self.data['Log_Returns'] = np.log(self.data['Price'] / self.data['Price'].shift(1))
        self.data['Log_Returns'] = self.data['Log_Returns'].fillna(0)
        return self.data
    
    def plot_price_series(self):
        """
        Visualize raw price series over time.
        """
        plt.figure(figsize=(16, 6))
        plt.plot(self.data['Date'], self.data['Price'], linewidth=1.5, color='darkblue')
        plt.title('Brent Crude Oil Prices (1987-2022)', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Price (USD per barrel)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('01_brent_price_series.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Price series plot saved')
    
    def plot_log_returns(self):
        """
        Visualize log returns and volatility clustering.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Log returns
        axes[0].plot(self.data['Date'], self.data['Log_Returns'], linewidth=0.5, color='darkgreen')
        axes[0].set_title('Daily Log Returns of Brent Oil Prices', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Log Return')
        axes[0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = self.data['Log_Returns'].rolling(window=30).std()
        axes[1].plot(self.data['Date'], rolling_vol, linewidth=1, color='darkred')
        axes[1].set_title('30-Day Rolling Volatility', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_xlabel('Year')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_log_returns_volatility.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Log returns and volatility plot saved')
    
    def summary_statistics(self):
        """
        Calculate descriptive statistics.
        """
        self.logger.info('Calculating summary statistics')
        
        stats = {
            'Price': {
                'Mean': self.data['Price'].mean(),
                'Median': self.data['Price'].median(),
                'Std Dev': self.data['Price'].std(),
                'Min': self.data['Price'].min(),
                'Max': self.data['Price'].max(),
                'Range': self.data['Price'].max() - self.data['Price'].min(),
                'CV': (self.data['Price'].std() / self.data['Price'].mean()) * 100
            },
            'Log_Returns': {
                'Mean': self.data['Log_Returns'].mean(),
                'Median': self.data['Log_Returns'].median(),
                'Std Dev': self.data['Log_Returns'].std(),
                'Min': self.data['Log_Returns'].min(),
                'Max': self.data['Log_Returns'].max(),
                'Skewness': self.data['Log_Returns'].skew(),
                'Kurtosis': self.data['Log_Returns'].kurtosis()
            }
        }
        
        print('\nTIME SERIES SUMMARY STATISTICS')
        print('='*60)
        print('\nPRICE STATISTICS:')
        for key, value in stats['Price'].items():
            if key == 'CV':
                print(f'  {key}: {value:.2f}%')
            else:
                print(f'  {key}: {value:.4f}')
        
        print('\nLOG RETURNS STATISTICS:')
        for key, value in stats['Log_Returns'].items():
            print(f'  {key}: {value:.6f}')
        
        return stats

class StationarityTester:
    """
    Test for stationarity using ADF and KPSS tests.
    """
    
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def adf_test(self, series, name):
        """
        Perform Augmented Dickey-Fuller test.
        Null hypothesis: unit root (non-stationary)
        """
        self.logger.info(f'Running ADF test on {name}')
        result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'name': name,
            'test': 'ADF',
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'stationary': result[1] < 0.05
        }
    
    def kpss_test(self, series, name):
        """
        Perform KPSS test.
        Null hypothesis: stationarity
        """
        self.logger.info(f'Running KPSS test on {name}')
        result = kpss(series.dropna(), regression='c')
        
        return {
            'name': name,
            'test': 'KPSS',
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[3],
            'stationary': result[1] > 0.05
        }
    
    def perform_tests(self):
        """
        Perform all stationarity tests.
        """
        results = []
        
        # Test on price series
        results.append(self.adf_test(self.data['Price'], 'Price Series'))
        results.append(self.kpss_test(self.data['Price'], 'Price Series'))
        
        # Test on log returns
        results.append(self.adf_test(self.data['Log_Returns'], 'Log Returns'))
        results.append(self.kpss_test(self.data['Log_Returns'], 'Log Returns'))
        
        return results
    
    def display_results(self, results):
        """
        Display test results in readable format.
        """
        print('\nSTATIONARITY TEST RESULTS')
        print('='*80)
        
        for result in results:
            print(f"\n{result['name']} - {result['test']} Test")
            print('-'*60)
            print(f"  Test Statistic: {result['test_statistic']:.6f}")
            print(f"  P-value: {result['p_value']:.6f}")
            print(f"  Stationary (α=0.05): {result['stationary']}")
            
            if result['test'] == 'ADF':
                print(f"  Interpretation: Reject H0 (non-stationary) at 5% level" if result['stationary'] 
                      else f"  Interpretation: Fail to reject H0 - likely non-stationary")
            else:
                print(f"  Interpretation: Fail to reject H0 (stationary) at 5% level" if result['stationary']
                      else f"  Interpretation: Reject H0 - likely non-stationary")
        
        print('\n' + '='*80)
        print('\nSUMMARY FOR MODELING:')
        print('  • Price Series: Non-stationary (as expected for prices))')
        print('  • Log Returns: Stationary (suitable for modeling)')
        print('  → Recommend using log returns or differencing in change point model')