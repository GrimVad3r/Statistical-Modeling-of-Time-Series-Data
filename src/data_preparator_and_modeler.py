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

class DataPreparator:
    """
    Modular class for data preparation in change point analysis.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)
        self.df = None
    
    def load_and_prepare(self):
        """
        Load data and perform preprocessing.
        """
        try:
            self.logger.info(f'Loading data from {self.filepath}')
            self.df = pd.read_csv(self.filepath)
            
            # Convert date and price
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
            
            # Sort and reset index
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            
            # Remove any NaN values
            self.df = self.df.dropna()
            
            self.logger.info(f'Data prepared: {len(self.df)} records from {self.df["Date"].min().date()} to {self.df["Date"].max().date()}')
            return self.df
        except Exception as e:
            self.logger.error(f'Error loading data: {str(e)}')
            raise
    
    def prepare_for_modeling(self):
        """
        Prepare data specifically for change point modeling.
        """
        self.logger.info('Preparing data for Bayesian modeling')
        
        # Calculate returns (alternative to using prices directly)
        self.df['Returns'] = self.df['Price'].pct_change() * 100
        self.df['Log_Returns'] = np.log(self.df['Price'] / self.df['Price'].shift(1))
        
        # Fill NaN from differencing
        self.df['Returns'] = self.df['Returns'].fillna(0)
        self.df['Log_Returns'] = self.df['Log_Returns'].fillna(0)
        
        # Create time index
        self.df['Time_Index'] = np.arange(len(self.df))
        
        self.logger.info('Data prepared for modeling with returns calculated')
        return self.df
class BayesianChangePointModel:
    """
    Build and fit a Bayesian change point model using PyMC.
    """
    
    def __init__(self, data, variable='Price'):
        self.data = data.copy()
        self.variable = variable
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.trace = None
        self.posterior = None
        
        self.logger.info(f'Initializing Bayesian Change Point Model for {variable}')
    
    def build_model(self):
        """
        Build the PyMC model with change point detection.
        Model structure:
        - Prior for change point (tau): Discrete uniform over all time indices
        - Prior for mean before change point (mu1): Normal distribution
        - Prior for mean after change point (mu2): Normal distribution
        - Prior for sigma: Exponential (half-normal would also work)
        - Likelihood: Normal distribution with switching mean
        """
        self.logger.info('Building PyMC model')
        
        # Extract price data and standardize
        price_data = self.data[self.variable].values
        n = len(price_data)
        
        # Standardize for better sampling
        price_mean = price_data.mean()
        price_std = price_data.std()
        price_standardized = (price_data - price_mean) / price_std
        
        self.logger.info(f'Building model with {n} observations')
        self.logger.info(f'Price data - Mean: {price_mean:.2f}, Std: {price_std:.2f}')
        
        with pm.Model() as model:
            # Priors
            # Change point: discrete uniform over all time points
            tau = pm.DiscreteUniform('tau', lower=1, upper=n-2)
            
            # Means before and after change point
            mu1 = pm.Normal('mu1', mu=0, sigma=1)
            mu2 = pm.Normal('mu2', mu=0, sigma=1)
            
            # Standard deviation (same for both regimes)
            sigma = pm.Exponential('sigma', lam=1)
            
            # Switch function to select mean based on time
            idx = np.arange(n)
            mu = pm.math.switch(tau >= idx, mu1, mu2)
            
            # Likelihood
            likelihood = pm.Normal('obs', mu=mu, sigma=sigma, observed=price_standardized)
        
        self.model = model
        self.logger.info('Model structure created successfully')
        return self.model
    
    def fit_model(self, draws=2000, tune=1000, cores=2, chains=2, random_seed=42):
        """
        Fit the model using MCMC sampling.
        """
        self.logger.info(f'Starting MCMC sampling: {draws} draws, {tune} tuning steps, {chains} chains')
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                cores=cores,
                chains=chains,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=True,
                target_accept=0.9
            )
        
        self.posterior = self.trace.posterior
        self.logger.info('MCMC sampling completed')
        return self.trace
    
    def check_convergence(self):
        """
        Check model convergence using diagnostic metrics.
        """
        self.logger.info('Checking convergence diagnostics')
        
        summary = az.summary(self.trace, var_names=['tau', 'mu1', 'mu2', 'sigma'])
        
        print('\nMODEL SUMMARY - CONVERGENCE DIAGNOSTICS')
        print('='*80)
        print(summary)
        print('\nKey Metrics Interpretation:')
        print('  • r_hat: Should be < 1.01 (close to 1.0 indicates convergence)')
        print('  • ess_bulk: Effective sample size (higher is better)')
        print('  • ess_tail: Effective sample size for tail (higher is better)')
        print('='*80)
        
        return summary
    
    def plot_trace(self):
        """
        Plot trace plots to assess mixing and convergence.
        """
        self.logger.info('Creating trace plots')
        
        az.plot_trace(self.trace, var_names=['tau', 'mu1', 'mu2', 'sigma'])
        plt.tight_layout()
        plt.savefig('./figures/task2_03_trace_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Trace plots saved')
class ChangePointInterpreter:
    """
    Extract, analyze, and interpret change point model results.
    """
    
    def __init__(self, model, trace, data):
        self.model = model
        self.trace = trace
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def extract_change_point(self):
        """
        Extract change point posterior samples and statistics.
        """
        self.logger.info('Extracting change point from posterior')
        
        tau_samples = self.trace.posterior['tau'].values.flatten()
        
        # Calculate statistics
        tau_mean = np.mean(tau_samples)
        tau_median = np.median(tau_samples)
        tau_std = np.std(tau_samples)
        tau_hdi = az.hdi(self.trace, var_names=['tau'])['tau'].values
        
        # Convert to date
        cp_date = self.data.iloc[int(tau_median)]['Date']
        
        results = {
            'tau_mean': tau_mean,
            'tau_median': tau_median,
            'tau_std': tau_std,
            'tau_hdi': tau_hdi,
            'cp_date': cp_date,
            'tau_samples': tau_samples
        }
        
        self.logger.info(f'Change point detected at index {tau_median} ({cp_date.date()})')
        return results
    
    def extract_parameters(self):
        """
        Extract parameter estimates before and after change point.
        """
        self.logger.info('Extracting model parameters')
        
        mu1_samples = self.trace.posterior['mu1'].values.flatten()
        mu2_samples = self.trace.posterior['mu2'].values.flatten()
        sigma_samples = self.trace.posterior['sigma'].values.flatten()
        
        # Denormalize prices
        price_mean = self.data['Price'].mean()
        price_std = self.data['Price'].std()
        
        mu1_denorm = mu1_samples * price_std + price_mean
        mu2_denorm = mu2_samples * price_std + price_mean
        sigma_denorm = sigma_samples * price_std
        
        parameters = {
            'mu1_mean': np.mean(mu1_denorm),
            'mu1_hdi': az.hdi(self.trace, var_names=['mu1'])['mu1'].values * price_std + price_mean,
            'mu2_mean': np.mean(mu2_denorm),
            'mu2_hdi': az.hdi(self.trace, var_names=['mu2'])['mu2'].values * price_std + price_mean,
            'sigma_mean': np.mean(sigma_denorm),
            'mu1_samples': mu1_denorm,
            'mu2_samples': mu2_denorm,
            'sigma_samples': sigma_denorm
        }
        
        self.logger.info(f'Mean before change point: ${parameters["mu1_mean"]:.2f}')
        self.logger.info(f'Mean after change point: ${parameters["mu2_mean"]:.2f}')
        
        return parameters
    
    def calculate_impact(self, change_point_results, parameters):
        """
        Quantify the impact of the detected change point.
        """
        self.logger.info('Calculating impact of change point')
        
        mu1 = parameters['mu1_mean']
        mu2 = parameters['mu2_mean']
        
        absolute_change = mu2 - mu1
        percent_change = (absolute_change / mu1) * 100 if mu1 != 0 else 0
        
        impact = {
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            'direction': 'Increase' if absolute_change > 0 else 'Decrease',
            'magnitude': 'Large' if abs(percent_change) > 50 else 'Moderate' if abs(percent_change) > 20 else 'Small'
        }
        
        return impact
    
    def display_results(self, cp_results, param_results, impact):
        """
        Display comprehensive results.
        """
        print('\n' + '='*80)
        print('BAYESIAN CHANGE POINT ANALYSIS RESULTS')
        print('='*80)
        
        print(f"\nCHANGE POINT DETECTION:")
        print('-'*60)
        print(f"  Detected Date: {cp_results['cp_date'].date()}")
        print(f"  Time Index: {int(cp_results['tau_median'])} days from start")
        print(f"  95% Credible Interval: Days {int(cp_results['tau_hdi'][0])} to {int(cp_results['tau_hdi'][1])}")
        print(f"  Certainty: 95% CI spans {int(cp_results['tau_hdi'][1] - cp_results['tau_hdi'][0])} days")
        
        print(f"\nPRICE PARAMETERS:")
        print('-'*60)
        print(f"  Mean Price BEFORE Change Point: ${param_results['mu1_mean']:.2f}/barrel")
        print(f"  95% CI: ${param_results['mu1_hdi'][0]:.2f} - ${param_results['mu1_hdi'][1]:.2f}")
        print(f"\n  Mean Price AFTER Change Point: ${param_results['mu2_mean']:.2f}/barrel")
        print(f"  95% CI: ${param_results['mu2_hdi'][0]:.2f} - ${param_results['mu2_hdi'][1]:.2f}")
        print(f"\n  Volatility (Sigma): ${param_results['sigma_mean']:.2f}/barrel")
        
        print(f"\nIMPACT QUANTIFICATION:")
        print('-'*60)
        print(f"  Absolute Change: ${impact['absolute_change']:.2f}/barrel")
        print(f"  Percentage Change: {impact['percent_change']:.2f}%")
        print(f"  Direction: {impact['direction']}")
        print(f"  Magnitude: {impact['magnitude']}")
        
        print('\n' + '='*80)

class EventAssociator:
    """
    Associate detected change points with historical events.
    """
    
    def __init__(self, data, events_file='major_events.csv'):
        self.data = data
        self.logger = logging.getLogger(__name__)
        
        try:
            self.events_df = pd.read_csv(events_file)
            self.events_df['date'] = pd.to_datetime(self.events_df['date'])
            self.logger.info(f'Loaded {len(self.events_df)} events from {events_file}')
        except FileNotFoundError:
            self.logger.warning(f'Events file {events_file} not found, creating sample events')
            self.events_df = pd.DataFrame()
    
    def find_nearby_events(self, cp_date, window_days=60):
        """
        Find events that occurred near the detected change point.
        """
        self.logger.info(f'Finding events within {window_days} days of {cp_date.date()}')
        
        if len(self.events_df) == 0:
            return pd.DataFrame()
        
        time_diff = (self.events_df['date'] - cp_date).dt.total_seconds() / (24 * 3600)
        nearby_mask = (time_diff.abs() <= window_days) & (time_diff >= -30)  # Event before CP is more relevant
        nearby_events = self.events_df[nearby_mask].copy()
        nearby_events['days_before_cp'] = -time_diff[nearby_mask].values
        
        return nearby_events.sort_values('days_before_cp')
    
    def create_event_impact_narrative(self, cp_date, cp_results, param_results, impact):
        """
        Create narrative explaining the change point and associated events.
        """
        nearby_events = self.find_nearby_events(cp_date, window_days=90)
        
        narrative = {
            'change_point_date': cp_date,
            'detected_date': cp_results['cp_date'],
            'confidence_interval': (cp_results['tau_hdi'][0], cp_results['tau_hdi'][1]),
            'price_before': param_results['mu1_mean'],
            'price_after': param_results['mu2_mean'],
            'absolute_change': impact['absolute_change'],
            'percent_change': impact['percent_change'],
            'nearby_events': nearby_events
        }
        
        return narrative
    
    def display_event_associations(self, narrative):
        """
        Display event associations in readable format.
        """
        print('\n' + '='*80)
        print('EVENT ASSOCIATION AND CAUSAL INTERPRETATION')
        print('='*80)
        
        print(f"\nDetected Change Point: {narrative['detected_date'].date()}")
        print(f"95% Credible Interval: {narrative['detected_date'].date()}")
        print(f"\nPrice Shift Summary:")
        print(f"  Before: ${narrative['price_before']:.2f}/barrel")
        print(f"  After: ${narrative['price_after']:.2f}/barrel")
        print(f"  Change: ${narrative['absolute_change']:.2f} ({narrative['percent_change']:.2f}%)")
        
        if len(narrative['nearby_events']) > 0:
            print(f"\nNearby Events (within 90 days):")
            print('-'*80)
            for idx, event in narrative['nearby_events'].iterrows():
                print(f"\n  {event['event_name']} ({event['date'].date()})")
                print(f"    Category: {event['category']}")
                print(f"    Description: {event['description']}")
                print(f"    Days before change point: {event.get('days_before_cp', 'N/A')}")
        else:
            print(f"\n  No events found in major events dataset within 90-day window")
            print(f"  This may indicate:")
            print(f"    1. Change point driven by gradual market forces")
            print(f"    2. Small-scale events not captured in major events list")
            print(f"    3. Anticipated market reaction ahead of formal event")
        
        print('\nIMPORTANT: Association does not imply causation!')
        print('Temporal correlation may reflect:")
        print('  • Market anticipation of known events')
        print('  • Delayed market reaction to earlier shocks')
        print('  • Coincidental timing with unrelated factors')
        print('='*80)