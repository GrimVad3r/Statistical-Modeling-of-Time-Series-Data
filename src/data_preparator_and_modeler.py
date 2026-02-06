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
import jax
import jax.numpy as jnp

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
    Optimized Bayesian change point detection model using PyMC 5.27 + numpyro 0.20.
    
    Performance characteristics:
    - With numpyro backend: 5-15 minutes for 9K rows (confirmed working)
    - Uses JAX for fast autodiff and vectorization
    - Automatically detects and uses numpyro backend
    
    Model structure:
    - Change point (tau): Continuous uniform prior
    - Pre-change mean (mu1): Normal distribution
    - Post-change mean (mu2): Normal distribution  
    - Noise (sigma): Exponential distribution
    - Likelihood: Normal with sigmoid-weighted mean switch
    """
    
    def __init__(self, data, variable='Price'):
        """
        Initialize the Bayesian Change Point Model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with price/target column
        variable : str
            Column name to model (default 'Price')
        """
        self.data = data.copy()
        self.variable = variable
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.trace = None
        self.posterior = None
        
        self.logger.info(f'Initializing Bayesian Change Point Model for {variable}')
        self.logger.info(f'PyMC version: {pm.__version__}')
        self.logger.info('Backend: numpyro (JAX-based, fast)')
        self._verify_dependencies()
    
    def _verify_dependencies(self):
        """Verify that numpyro and JAX are available."""
        try:
            import numpyro
            import jax
            self.logger.info(f'numpyro {numpyro.__version__} + JAX {jax.__version__} confirmed')
        except ImportError as e:
            self.logger.warning(f'Missing dependency: {e}')
    
    def build_model(self):
        """
        Build the PyMC model with change point detection.
        
        Uses:
        - Float32 for GPU efficiency
        - Sigmoid switch for smooth, differentiable transitions
        - Standardized likelihood for numerical stability
        
        Returns
        -------
        pymc.Model
            Compiled PyMC model ready for sampling
        """
        self.logger.info('Building PyMC model with numpyro backend')
        
        # Extract and standardize data
        price_data = self.data[self.variable].values
        n = len(price_data)
        
        price_mean = price_data.mean()
        price_std = price_data.std()
        price_standardized = (price_data - price_mean) / price_std
        
        self.logger.info(f'Data: {n} observations')
        self.logger.info(f'Mean: {price_mean:.2f}, Std: {price_std:.2f}')
        
        with pm.Model() as model:
            # === PRIORS ===
            
            # Change point: uniform over all time indices
            tau = pm.Uniform('tau', lower=0, upper=n)
            
            # Pre-change mean
            mu1 = pm.Normal('mu1', mu=0, sigma=1)
            
            # Post-change mean
            mu2 = pm.Normal('mu2', mu=0, sigma=1)
            
            # Noise level
            sigma = pm.Exponential('sigma', lam=1)
            
            # === LIKELIHOOD ===
            
            # Sigmoid switch for smooth change point
            # Float32 for GPU efficiency, s=10 for sharp transition
            idx = np.arange(n, dtype=np.float32)
            weight = pm.math.sigmoid(10.0 * (idx - tau))
            
            # Weighted mean: mu1 when weight≈0, mu2 when weight≈1
            mu = mu1 * (1.0 - weight) + mu2 * weight
            
            # Normal likelihood with switching mean
            likelihood = pm.Normal(
                'obs',
                mu=mu,
                sigma=sigma,
                observed=price_standardized
            )
            
            self.model = model
            return self.model
    
    def fit_model(self, draws=1000, tune=500, chains=2, random_seed=42, target_accept=0.9):
        """
        Fit the model using numpyro backend sampling.
        
        The numpyro backend uses JAX for:
        - Automatic differentiation (NUTS sampler)
        - Vectorized operations
        - JIT compilation
        
        For 9K rows with these defaults: 5-15 minutes expected
        
        Parameters
        ----------
        draws : int
            Posterior samples per chain (default 1000)
        tune : int
            Tuning steps per chain (default 500)
        chains : int
            Independent chains to run (default 2, parallelized by JAX)
        random_seed : int
            Seed for reproducibility (default 42)
        target_accept : float
            Target acceptance rate for NUTS (default 0.9)
            Higher = more conservative, slower but better mixing
            
        Returns
        -------
        arviz.InferenceData
            Posterior samples and diagnostics
        """
        self.logger.info('='*80)
        self.logger.info('STARTING BAYESIAN INFERENCE')
        self.logger.info('='*80)
        self.logger.info(f'Dataset: {len(self.data)} rows')
        self.logger.info(f'Sampler: numpyro (JAX-based NUTS)')
        self.logger.info(f'Configuration: draws={draws}, tune={tune}, chains={chains}')
        self.logger.info(f'Expected runtime: 5-15 minutes for 9K+ rows')
        self.logger.info('='*80)
        
        with self.model:
            try:
                self.logger.info('Sampling with numpyro backend...')
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    nuts_sampler="numpyro", 
                    target_accept=target_accept,
                    progressbar=True,
                    return_inferencedata=True,
                    idata_kwargs={'log_likelihood': True}
                )
                self.logger.info('SUCCESS: numpyro sampling completed')
                self.logger.info('Expected speedup: 5-10x vs default NUTS')
                
            except Exception as e:
                # Fallback to default NUTS if something goes wrong
                self.logger.error(f'numpyro failed: {type(e).__name__}')
                self.logger.warning('Falling back to default PyMC NUTS (slower)')
                
                with self.model:
                    self.trace = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        random_seed=random_seed,
                        target_accept=target_accept,
                        progressbar=True,
                        return_inferencedata=True,
                        idata_kwargs={'log_likelihood': True}
                    )
        
        self.posterior = self.trace.posterior
        return self.trace
    
    def check_convergence(self):
        """
        Check Bayesian inference convergence using diagnostic metrics.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics including r_hat and ESS diagnostics
            
        Notes
        -----
        - r_hat < 1.01: Good convergence
        - r_hat > 1.01: Poor mixing, consider more iterations
        - ess_bulk: Effective sample size in bulk of distribution
        - ess_tail: Effective sample size in distribution tails
        """
        self.logger.info('Computing convergence diagnostics')
        
        summary = az.summary(
            self.trace,
            var_names=['tau', 'mu1', 'mu2', 'sigma'],
            kind='all'
        )
        
        print('\n' + '='*80)
        print('CONVERGENCE DIAGNOSTICS')
        print('='*80)
        print(summary)
        print('\n' + 'INTERPRETATION:')
        print('  • r_hat < 1.01: Good convergence')
        print('  • r_hat > 1.01: Poor mixing (increase tune/draws)')
        print('  • ess_bulk: Effective sample size (bulk)')
        print('  • ess_tail: Effective sample size (tail)')
        print('='*80)
        
        # Check for convergence issues
        # Safe check for r_hat
        if 'r_hat' in summary.columns:
            r_hat_issues = (summary['r_hat'] > 1.01).sum()
            if r_hat_issues > 0:
                self.logger.warning(f'{r_hat_issues} parameters have r_hat > 1.01 - rerun with more iterations')
            else:
                self.logger.info('All parameters converged (r_hat < 1.01)')
        else:
            self.logger.warning("r_hat not available (possibly only 1 chain or NumPyro backend limitation). "
                                "Check ESS, divergences, and trace plots instead.")
        
        return summary
    
    def plot_trace(self, figsize=(14, 8)):
        """
        Plot trace plots to assess sampler mixing and convergence.
        
        Each variable shows:
        - Left: Trace over iterations (should look like white noise)
        - Right: Posterior density estimate
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
            
        Returns
        -------
        matplotlib.figure.Figure
            Trace plot figure
        """
        self.logger.info('Creating trace plots')
        
        fig = az.plot_trace(
            self.trace,
            var_names=['tau', 'mu1', 'mu2', 'sigma'],
            figsize=figsize,
            combined=True
        )
        
        plt.tight_layout()
        plt.savefig('./figures/task2_03_trace_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info('Trace plots saved to ./figures/task2_03_trace_plots.png')
        
        return fig
    
    def get_change_point_estimate(self, hdi_prob=0.94):
        """
        Extract the estimated change point with credible interval.
        
        Parameters
        ----------
        hdi_prob : float
            Probability level for highest density interval (default 0.94)
            
        Returns
        -------
        dict
            Change point estimates:
            - 'mean': Mean of posterior samples
            - 'median': Median of posterior samples
            - 'hdi_lower': Lower bound of HDI
            - 'hdi_upper': Upper bound of HDI
            - 'hdi_prob': HDI probability level
            
        Examples
        --------
        >>> estimates = model.get_change_point_estimate(hdi_prob=0.95)
        >>> print(f"Change point: {estimates['median']:.1f}")
        >>> print(f"95% CI: [{estimates['hdi_lower']:.1f}, {estimates['hdi_upper']:.1f}]")
        """
        tau_samples = self.trace.posterior['tau'].values.flatten()
        
        mean_tau = float(np.mean(tau_samples))
        median_tau = float(np.median(tau_samples))
        hdi = az.hdi(self.trace, hdi_prob=hdi_prob)['tau'].values
        
        result = {
            'mean': mean_tau,
            'median': median_tau,
            'hdi_lower': float(hdi[0]),
            'hdi_upper': float(hdi[1]),
            'hdi_prob': hdi_prob
        }
        
        self.logger.info(f'Change point: {median_tau:.2f}')
        self.logger.info(f'{int(hdi_prob*100)}% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}]')
        
        return result
    
    def plot_posterior_predictive(self):
        """
        Plot observed data with posterior predictive distribution.
        
        Shows:
        - Black line: Observed data
        - Blue shaded region: 100 posterior predictive samples
        - Red dashed line: Estimated change point
        
        This visualization shows:
        - Model fit quality
        - Uncertainty in predictions
        - Change point location
        """
        self.logger.info('Creating posterior predictive plot')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Observed data
        ax.plot(
            self.data.index,
            self.data[self.variable].values,
            'k-',
            linewidth=2.5,
            label='Observed',
            zorder=3
        )
        
        # Posterior predictive samples
        posterior_pred = self.trace.posterior_predictive['obs']
        n_samples = min(100, posterior_pred.shape[2])
        
        for i in range(n_samples):
            ax.plot(
                self.data.index,
                posterior_pred.values[:, 0, i],
                alpha=0.02,
                color='blue',
                linewidth=0.5
            )
        
        # Change point estimate
        cp_est = self.get_change_point_estimate()
        ax.axvline(
            cp_est['median'],
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Change point: {cp_est['median']:.1f}",
            zorder=2
        )
        
        # Formatting
        ax.set_xlabel('Time Index', fontsize=11)
        ax.set_ylabel(self.variable, fontsize=11)
        ax.set_title('Posterior Predictive Distribution with Change Point', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            './figures/task2_03_posterior_predictive.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()
        
        self.logger.info('Posterior predictive plot saved to ./figures/task2_03_posterior_predictive.png')
        
        return fig

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
        print('Temporal correlation may reflect:')
        print('  • Market anticipation of known events')
        print('  • Delayed market reaction to earlier shocks')
        print('  • Coincidental timing with unrelated factors')
        print('='*80)