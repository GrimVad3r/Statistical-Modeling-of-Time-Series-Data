# BRENT OIL PRICES CHANGE POINT ANALYSIS
## Complete Step-by-Step Instructions with Code Modularization & Logging

**Date**: 04 Feb - 10 Feb 2026  
**Challenge**: Change Point Analysis and Statistical Modeling of Time Series Data  
**Organization**: Birhan Energies  

---

## TABLE OF CONTENTS

1. [Overview](#overview)
2. [Task 1: Laying the Foundation for Analysis](#task-1)
3. [Task 2: Change Point Modeling and Insight Generation](#task-2)
4. [Task 3: Interactive Dashboard Development](#task-3)
5. [Code Modularization Principles](#code-modularization)
6. [Logging Strategy](#logging-strategy)
7. [File Deliverables](#file-deliverables)

---

## OVERVIEW

This challenge requires analyzing how major political and economic events affect Brent oil prices using Bayesian change point detection. The work is organized into three tasks with increasing complexity.

### Key Learning Outcomes:
- Change Point Analysis & Interpretation
- Bayesian Inference and MCMC
- PyMC model building and interpretation
- Dashboard development with Flask + React
- Data storytelling and communication

### Data:
- **Dataset**: BrentOilPrices.csv (9,012 daily records from May 1987 to September 2022)
- **Columns**: Date (day-month-year format), Price (USD per barrel)

---

## TASK 1: LAYING THE FOUNDATION FOR ANALYSIS

### Objective
Define the data analysis workflow and develop thorough understanding of the model and data.

### Step 1.1: Define the Data Analysis Workflow

**What to do:**
1. Outline 8-phase analysis pipeline (Data Loading → Preprocessing → EDA → Modeling → Results → Events → Dashboard → Reporting)
2. Document each phase's tasks, dependencies, and tools
3. Create a structured workflow diagram

**Code Implementation:**
```python
class DataAnalysisWorkflow:
    def document_workflow(self):
        # Define 8 phases with tasks, dependencies, and tools
        # Document dependencies between phases
        # Create visual representation
```

**Logging Points:**
- Log when workflow is documented
- Log phase definitions and dependencies
- Log completion of workflow planning

### Step 1.2: Research and Compile Major Events

**What to do:**
1. Research 10-15 major events (1987-2022):
   - Geopolitical conflicts (Iraq, Kuwait, Libya, etc.)
   - OPEC policy decisions
   - Economic crises (Asian crisis 1997, Financial crisis 2008, COVID 2020)
   - Sanctions and trade wars
2. Create structured CSV with: Date, Event Name, Category, Description
3. Include approximate dates for major shifts

**Key Events to Include:**
- 1990-08-02: Iraq Invasion of Kuwait
- 1991-01-17: Gulf War Begins
- 1997-07-02: Asian Financial Crisis
- 2001-09-11: September 11 Attacks
- 2003-03-20: Iraq War Begins
- 2008-07-11: Oil Prices Peak (~$145)
- 2008-09-15: Lehman Brothers Collapse
- 2011-03-15: Libyan Civil War
- 2014-06-01: Oil Price Decline Begins
- 2014-11-27: OPEC Abandons Production Cuts
- 2016-02-11: Oil Prices Hit Bottom (~$26)
- 2016-11-30: OPEC Production Cuts Announced
- 2020-02-01: COVID-19 Pandemic
- 2020-04-20: Oil Prices Turn Negative

**Code Implementation:**
```python
class EventCompiler:
    def compile_events(self):
        # Create list of 15+ events with structured format
        # Export to CSV
        # Validate dates and descriptions
    
    def export_to_csv(self, filepath):
        # Save events with proper formatting
```

**Deliverable**: major_events.csv

### Step 1.3: Analyze Time Series Properties

**What to do:**
1. Load and validate Brent oil price data
2. Calculate key statistics:
   - Mean, median, standard deviation, min, max
   - Range and coefficient of variation
3. Analyze time series properties:
   - **Trend**: Long-term upward/downward movement
   - **Seasonality**: Recurring patterns within years
   - **Volatility**: Clustering of price changes
4. Calculate log returns for stationarity analysis
5. Plot visualizations

**Code Implementation:**
```python
class TimeSeriesAnalyzer:
    def calculate_log_returns(self):
        # Calculate log(price_t / price_{t-1})
        # Store in dataframe
    
    def plot_price_series(self):
        # Line plot of prices over time
        # Mark major events if available
    
    def plot_log_returns(self):
        # Scatter/line plot of returns
        # Show volatility clustering
    
    def summary_statistics(self):
        # Calculate descriptive statistics
        # Display in table format
```

**Logging Points:**
- Log data loading (shape, date range)
- Log each statistical calculation
- Log visualization creation
- Log warnings about missing values

**Deliverables**:
- 01_brent_price_series.png
- 02_log_returns_volatility.png
- Summary statistics table

### Step 1.4: Stationarity Testing

**What to do:**
1. Perform Augmented Dickey-Fuller (ADF) test on price series
2. Perform KPSS test for confirmation
3. Repeat tests on log returns
4. Interpret results

**Code Implementation:**
```python
class StationarityTester:
    def adf_test(self, series, name):
        # Run ADF test
        # Return test statistic and p-value
        # Interpret (reject H0 if p < 0.05)
    
    def kpss_test(self, series, name):
        # Run KPSS test
        # Return test statistic and p-value
        # Interpret (fail to reject H0 if p > 0.05)
    
    def display_results(self, results):
        # Create comparison table
        # Show conclusions
```

**Results Expected:**
- Price Series: Non-stationary (p > 0.05 in ADF)
- Log Returns: Stationary (p < 0.05 in ADF)
- Conclusion: Use log returns or differencing in model

### Step 1.5: Document Assumptions and Limitations

**What to do:**
1. List 6+ key assumptions:
   - Data quality assumption
   - Model specification (single change point)
   - Statistical independence
   - Normality of returns
   - Event timing accuracy
   - Causal attribution possibility

2. Document 6+ limitations:
   - Temporal scope (1987-2022 only)
   - Univariate analysis (only prices)
   - Event data quality/timing
   - Model simplicity (constant variance)
   - MCMC sampling uncertainty
   - Confounding factors

3. Explain correlation vs causation distinction:
   - Correlation: Temporal association
   - Causation: Causal mechanism
   - Requirements for causation (5 criteria)

**Code Implementation:**
```python
class AssumptionsAndLimitations:
    def document(self):
        # Create structured documentation
        # Include risk levels
        # Explain implications
    
    def export_to_file(self, doc, filepath):
        # Save professional documentation
        # Formatted for readability
```

**Deliverable**: assumptions_and_limitations.txt

### Step 1.6: Data Loading Module

**What to do:**
1. Create modular DataLoader class
2. Implement load_data() method
3. Implement preprocess_data() method
4. Implement validate_data() method
5. Add comprehensive logging

**Code Implementation:**
```python
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)
        self.df = None
    
    def load_data(self):
        # Read CSV with error handling
        # Log shape and contents
    
    def preprocess_data(self):
        # Convert Date to datetime
        # Sort by date
        # Remove missing values
        # Log transformations
    
    def validate_data(self):
        # Check for nulls
        # Verify date range
        # Confirm data types
        # Return validation report
```

**Logging Points:**
- File access and load success
- Data shape and structure
- Date range discovery
- Missing value handling
- Data type conversions
- Validation results

### Task 1 Deliverables

**Due**: Interim Submission - Sunday, 08 Feb 2026, 8:00 PM UTC

Checklist:
- [ ] major_events.csv (15+ events with dates and descriptions)
- [ ] Assumptions and Limitations document (1-2 pages)
- [ ] Time series visualizations (2-3 plots)
- [ ] Summary statistics calculated and displayed
- [ ] Stationarity test results documented
- [ ] Jupyter notebook with complete Task 1 code (JSON format)
- [ ] task_1_analysis.log file with all logging outputs

---

## TASK 2: CHANGE POINT MODELING AND INSIGHT GENERATION

### Objective
Apply Bayesian change point detection to identify structural breaks in Brent oil prices.

### Step 2.1: Prepare Data for Modeling

**What to do:**
1. Load cleaned data from Task 1
2. Create time index (0 to n-1)
3. Standardize price data (for better MCMC sampling)
4. Calculate returns (optional alternative to prices)
5. Split into potential train/test sets if needed

**Code Implementation:**
```python
class DataPreparator:
    def prepare_for_modeling(self):
        # Standardize prices: (price - mean) / std
        # Create time indices
        # Calculate returns if needed
        # Log all transformations
    
    def get_data_summary(self):
        # Return metadata for model configuration
        # Include number of observations
        # Include mean and std for denormalization
```

**Why Standardization?**
MCMC samplers mix better with standardized data. We denormalize results after sampling.

### Step 2.2: Build Bayesian Change Point Model

**Model Structure:**
```
Priors:
  tau ~ DiscreteUniform(1, n-2)         # Change point time index
  mu1 ~ Normal(0, 1)                    # Mean before change point
  mu2 ~ Normal(0, 1)                    # Mean after change point
  sigma ~ Exponential(1)                # Standard deviation (same for both regimes)

Likelihood:
  For each observation t:
    if t < tau: price_t ~ Normal(mu1, sigma)
    if t >= tau: price_t ~ Normal(mu2, sigma)

Switch mechanism:
  mu_t = pm.math.switch(tau >= t, mu1, mu2)
```

**Code Implementation:**
```python
class BayesianChangePointModel:
    def __init__(self, data, variable='Price'):
        self.data = data
        self.variable = variable
        self.model = None
        self.trace = None
    
    def build_model(self):
        # Extract and standardize data
        # Create PyMC model context
        # Define priors with pm.* distributions
        # Define switch function using pm.math.switch
        # Define likelihood with pm.Normal
        # Return compiled model
    
    def fit_model(self, draws=2000, tune=1000, cores=2, chains=2):
        # Run pm.sample() with specified parameters
        # Log sampling progress
        # Return trace object with posterior samples
        # Log completion time
```

**PyMC Model Details:**

```python
with pm.Model() as model:
    # Standardized data
    price_data_std = (price_data - price_mean) / price_std
    n = len(price_data_std)
    
    # Priors
    tau = pm.DiscreteUniform('tau', lower=1, upper=n-2)
    mu1 = pm.Normal('mu1', mu=0, sigma=1)
    mu2 = pm.Normal('mu2', mu=0, sigma=1)
    sigma = pm.Exponential('sigma', lam=1)
    
    # Switch mechanism
    idx = np.arange(n)
    mu = pm.math.switch(tau >= idx, mu1, mu2)
    
    # Likelihood
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=price_data_std)
    
    # Sample
    trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True)
```

**Logging Points:**
- Model structure created
- Data standardization parameters
- Prior specifications
- Sampling start and completion
- Sampling time
- Convergence warnings

### Step 2.3: Check Convergence Diagnostics

**What to do:**
1. Check r_hat values (should be < 1.01)
2. Examine effective sample size (ess_bulk, ess_tail)
3. Create trace plots to visualize MCMC chains
4. Look for good mixing (chains overlapping)
5. Check for autocorrelation

**Code Implementation:**
```python
class ConvergenceDiagnostics:
    def check_convergence(self, trace):
        # Calculate summary statistics
        summary = az.summary(trace, var_names=['tau', 'mu1', 'mu2', 'sigma'])
        
        # Check r_hat values
        r_hat_values = summary['r_hat']
        
        # Check effective sample sizes
        ess_bulk = summary['ess_bulk']
        ess_tail = summary['ess_tail']
        
        # Log results
        return summary
    
    def plot_trace(self, trace):
        # Use arviz to create trace plots
        # Show posterior distribution and chain traces side by side
        az.plot_trace(trace, var_names=['tau', 'mu1', 'mu2', 'sigma'])
```

**Interpretation:**
- r_hat < 1.01: Good convergence
- r_hat > 1.05: Chains haven't converged
- Solution: Increase tuning steps or adjust priors

### Step 2.4: Extract and Interpret Results

**What to do:**
1. Extract posterior samples for tau, mu1, mu2, sigma
2. Calculate point estimates (mean, median)
3. Calculate credible intervals (95% HDI)
4. Denormalize parameters back to original scale
5. Calculate impact metrics

**Code Implementation:**
```python
class ChangePointInterpreter:
    def extract_change_point(self, trace, data):
        # Get tau samples from trace
        tau_samples = trace.posterior['tau'].values.flatten()
        
        # Calculate statistics
        tau_median = np.median(tau_samples)
        tau_hdi = az.hdi(trace, var_names=['tau'])['tau'].values
        
        # Convert to date
        cp_date = data.iloc[int(tau_median)]['Date']
        
        # Log findings
        return {
            'tau_samples': tau_samples,
            'tau_median': tau_median,
            'tau_hdi': tau_hdi,
            'cp_date': cp_date
        }
    
    def extract_parameters(self, trace, data):
        # Get mu1, mu2, sigma samples
        mu1_samples = trace.posterior['mu1'].values.flatten()
        mu2_samples = trace.posterior['mu2'].values.flatten()
        sigma_samples = trace.posterior['sigma'].values.flatten()
        
        # Denormalize
        price_mean = data['Price'].mean()\n        price_std = data['Price'].std()
        mu1_denorm = mu1_samples * price_std + price_mean
        mu2_denorm = mu2_samples * price_std + price_mean
        
        return {
            'mu1_mean': np.mean(mu1_denorm),
            'mu1_hdi': az.hdi(trace, var_names=['mu1'])['mu1'].values * price_std + price_mean,
            'mu2_mean': np.mean(mu2_denorm),
            'mu2_hdi': az.hdi(trace, var_names=['mu2'])['mu2'].values * price_std + price_mean,
            'sigma_mean': np.mean(sigma_samples * price_std)
        }
    
    def calculate_impact(self, params):
        # Absolute change
        absolute_change = params['mu2_mean'] - params['mu1_mean']
        
        # Percent change
        percent_change = (absolute_change / params['mu1_mean']) * 100
        
        return {
            'absolute_change': absolute_change,
            'percent_change': percent_change,
            'direction': 'Increase' if absolute_change > 0 else 'Decrease'
        }
```

**Key Results to Report:**
- Change point date ± credible interval
- Mean price before change point with CI
- Mean price after change point with CI
- Absolute and percentage change
- Volatility (sigma)

### Step 2.5: Visualize Posterior Distributions

**What to do:**
1. Plot posterior distribution of tau (histogram + overlay on time series)
2. Plot posterior distributions of mu1, mu2, sigma
3. Show credible intervals
4. Compare before/after distributions
5. Create publication-quality figures

**Code Implementation:**
```python
class PosteriorVisualizer:
    def plot_change_point_posterior(self, tau_samples, data, cp_date):
        # Histogram of tau samples
        # Mark median and HDI
        # Overlay on price series with shaded CI region
        # Save figure
    
    def plot_parameter_posteriors(self, mu1_samples, mu2_samples, sigma_samples):
        # Create 4-panel figure
        # Panel 1: mu1 histogram with mean line
        # Panel 2: mu2 histogram with mean line
        # Panel 3: sigma histogram
        # Panel 4: Overlaid mu1 and mu2 for comparison
        # Save figure
```

### Step 2.6: Associate Change Points with Historical Events

**What to do:**
1. Load events CSV from Task 1
2. Find events within ±90 days of detected change point
3. Sort by temporal proximity
4. Create narrative linking change point to events
5. **Important**: Note that association ≠ causation

**Code Implementation:**
```python
class EventAssociator:
    def find_nearby_events(self, cp_date, events_df, window_days=60):
        # Calculate days from each event to change point
        # Filter to events within window (especially before)
        # Sort by temporal proximity
        # Return filtered events
    
    def create_event_impact_narrative(self, cp_date, results, params, impact):
        # Combine all results into cohesive narrative
        # Include quantified impacts
        # Add important caveats about causation
        # Log for documentation
```

**Narrative Structure:**
```
"Detected Change Point: [DATE]
With 95% credibility within [DATE ±N DAYS]

Price Shift: From $X before to $Y after (+/- Z%)

Nearby Events:
[Event 1] - [N days before change point]
[Event 2] - [N days before change point]

IMPORTANT: While these events show temporal association with the detected
change point, this does not establish causation. The price shift may reflect:
- Market anticipation of the event
- Delayed reaction to earlier shocks
- Coincidental timing with other factors
```

### Step 2.7: Advanced Extensions (Optional)

**For deeper analysis, consider:**
1. Multiple change points model
2. Regime-switching volatility
3. Incorporating exogenous variables (GDP, exchange rates)
4. Vector autoregression (VAR) model
5. Comparison with alternative models

### Task 2 Deliverables

**Due**: Final Submission - Tuesday, 10 Feb 2026, 8:00 PM UTC

Checklist:
- [ ] Complete Jupyter notebook with Task 2 code (JSON format)
- [ ] EDA visualizations (price, returns, volatility)
- [ ] Trace plots showing MCMC convergence
- [ ] Change point posterior distribution plot
- [ ] Parameter posterior distribution plots (4-panel)
- [ ] Convergence summary table (r_hat, ess, etc.)
- [ ] Event association narrative
- [ ] Quantified impact statements
- [ ] task_2_modeling.log file
- [ ] Model results saved (pickle or JSON)

---

## TASK 3: INTERACTIVE DASHBOARD DEVELOPMENT

### Objective
Build a Flask/React dashboard to visualize analysis results for stakeholders.

### Step 3.1: Design System Architecture

**Backend (Flask):**
- REST API with 5+ endpoints
- DataManager class for data loading
- Routes organized by concern
- Comprehensive error handling
- Logging for debugging

**Frontend (React):**
- Component-based architecture
- Recharts for visualizations
- Tailwind CSS for styling
- State management (Context API or Redux)
- Responsive design

**API Endpoints:**
1. `/api/v1/health` - Server health check
2. `/api/v1/prices` - Historical prices with date filtering
3. `/api/v1/change-point` - Change point analysis results
4. `/api/v1/events` - Major events listing
5. `/api/v1/statistics` - Summary statistics

### Step 3.2: Build Flask Backend

**File Structure:**
```
backend/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── data/
│   ├── BrentOilPrices.csv
│   ├── major_events.csv
│   └── model_results.pkl
├── routes/
│   ├── __init__.py
│   ├── prices.py            # Price endpoints
│   ├── changepoint.py       # Change point endpoints
│   └── events.py            # Events endpoints
└── utils/
    ├── __init__.py
    └── data_manager.py      # Data utilities
```

**Core Implementation:**

```python
# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.load_all_data()
    
    def load_all_data(self):
        """Load all required data on initialization."""
        try:
            self.price_data = pd.read_csv('data/BrentOilPrices.csv')
            self.events_data = pd.read_csv('data/major_events.csv')
            with open('data/model_results.pkl', 'rb') as f:
                self.model_results = pickle.load(f)
            self.logger.info('All data loaded successfully')
        except Exception as e:
            self.logger.error(f'Error loading data: {str(e)}')
            raise

data_manager = DataManager()

# API Endpoints
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/v1/prices', methods=['GET'])
def get_prices():
    \"\"\"Get historical prices with optional filtering.\"\"\"
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        prices = data_manager.price_data.copy()
        if start_date:
            prices = prices[prices['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            prices = prices[prices['Date'] <= pd.to_datetime(end_date)]
        
        logger.info(f'Returning {len(prices)} price records')
        return jsonify({'data': prices.to_dict('records'), 'count': len(prices)})
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/change-point', methods=['GET'])
def get_change_point():
    \"\"\"Get change point results.\"\"\"
    results = {
        'detected_date': data_manager.model_results['cp_date'].isoformat(),
        'mean_before': float(data_manager.model_results['mu1_mean']),
        'mean_after': float(data_manager.model_results['mu2_mean']),
        'absolute_change': float(data_manager.model_results['absolute_change']),
        'percent_change': float(data_manager.model_results['percent_change'])
    }
    return jsonify(results)

@app.route('/api/v1/events', methods=['GET'])
def get_events():
    \"\"\"Get major events.\"\"\"
    return jsonify({'data': data_manager.events_data.to_dict('records')})

@app.route('/api/v1/statistics', methods=['GET'])
def get_statistics():
    \"\"\"Get summary statistics.\"\"\"
    prices = data_manager.price_data['Price']
    stats = {
        'mean': float(prices.mean()),
        'median': float(prices.median()),
        'std_dev': float(prices.std()),
        'min': float(prices.min()),
        'max': float(prices.max())
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Step 3.3: Build React Frontend

**Key Components:**

```jsx
// Dashboard.jsx - Main component
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

const Dashboard = () => {
  const [priceData, setPriceData] = useState([]);
  const [changePoint, setChangePoint] = useState(null);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [prices, cp, evts, stat] = await Promise.all([
          axios.get('http://localhost:5000/api/v1/prices'),
          axios.get('http://localhost:5000/api/v1/change-point'),
          axios.get('http://localhost:5000/api/v1/events'),
          axios.get('http://localhost:5000/api/v1/statistics')
        ]);
        
        setPriceData(prices.data.data);
        setChangePoint(cp.data);
        setEvents(evts.data.data);
        setStats(stat.data);
        setLoading(false);
      } catch (error) {
        console.error('Error:', error);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <div className=\"text-center p-8\">Loading...</div>;

  return (
    <div className=\"min-h-screen bg-gray-50 p-8\">
      <h1 className=\"text-4xl font-bold mb-8\">Brent Oil Prices - Change Point Analysis</h1>
      
      {/* Summary Cards */}
      <div className=\"grid grid-cols-1 md:grid-cols-4 gap-4 mb-8\">
        <SummaryCard title=\"Mean Price\" value={`$${stats?.mean.toFixed(2)}`} />
        <SummaryCard title=\"Max Price\" value={`$${stats?.max.toFixed(2)}`} />
        <SummaryCard title=\"Min Price\" value={`$${stats?.min.toFixed(2)}`} />
        <SummaryCard title=\"Volatility\" value={`$${stats?.std_dev.toFixed(2)}`} />
      </div>

      {/* Price Chart */}
      <div className=\"bg-white p-6 rounded-lg shadow-lg mb-8\">
        <h2 className=\"text-2xl font-bold mb-4\">Historical Price Trends</h2>
        <ResponsiveContainer width=\"100%\" height={400}>
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray=\"3 3\" />
            <XAxis dataKey=\"Date\" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type=\"monotone\" dataKey=\"Price\" stroke=\"#2563eb\" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Change Point Results */}
      {changePoint && (
        <div className=\"bg-white p-6 rounded-lg shadow-lg mb-8\">
          <h2 className=\"text-2xl font-bold mb-4\">Change Point Analysis</h2>
          <div className=\"grid grid-cols-2 gap-4\">
            <div>
              <p className=\"text-gray-600\">Detected Date</p>
              <p className=\"text-xl font-bold\">{new Date(changePoint.detected_date).toLocaleDateString()}</p>
            </div>
            <div>
              <p className=\"text-gray-600\">Price Change</p>
              <p className=\"text-xl font-bold text-green-600\">${changePoint.absolute_change.toFixed(2)}</p>
            </div>
            <div>
              <p className=\"text-gray-600\">Mean Before</p>
              <p className=\"text-lg\">${changePoint.mean_before.toFixed(2)}</p>
            </div>
            <div>
              <p className=\"text-gray-600\">Mean After</p>
              <p className=\"text-lg\">${changePoint.mean_after.toFixed(2)}</p>
            </div>
          </div>
        </div>
      )}

      {/* Events */}
      <div className=\"bg-white p-6 rounded-lg shadow-lg\">
        <h2 className=\"text-2xl font-bold mb-4\">Major Events</h2>
        <div className=\"space-y-4\">
          {events.slice(0, 10).map((event, idx) => (
            <div key={idx} className=\"border-l-4 border-blue-500 pl-4\">
              <h3 className=\"font-semibold\">{event.event_name}</h3>
              <p className=\"text-sm text-gray-600\">{event.date}</p>
              <p className=\"text-sm\">{event.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const SummaryCard = ({ title, value }) => (
  <div className=\"bg-white p-4 rounded-lg shadow\">
    <p className=\"text-gray-600 text-sm\">{title}</p>
    <p className=\"text-2xl font-bold\">{value}</p>
  </div>
);

export default Dashboard;
```

### Step 3.4: Setup and Deployment

**Backend Setup:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
# Output: Running on http://localhost:5000
```

**Frontend Setup:**
```bash
# Create React app
npx create-react-app frontend
cd frontend

# Install packages
npm install recharts axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Start development server
npm start
# Output: App running on http://localhost:3000
```

### Step 3.5: Testing

**Backend Tests:**
```python
def test_health_endpoint():
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_prices_endpoint():
    response = client.get('/api/v1/prices')
    assert response.status_code == 200
    assert 'data' in response.json
    assert 'count' in response.json

def test_change_point_endpoint():
    response = client.get('/api/v1/change-point')
    assert response.status_code == 200
    assert 'detected_date' in response.json
    assert 'mean_before' in response.json
```

**Frontend Tests:**
```javascript
import { render, screen } from '@testing-library/react';
import Dashboard from './Dashboard';

test('renders dashboard title', () => {
  render(<Dashboard />);
  expect(screen.getByText(/Brent Oil Prices/i)).toBeInTheDocument();
});

test('displays summary cards', () => {
  render(<Dashboard />);
  expect(screen.getByText(/Mean Price/i)).toBeInTheDocument();
});
```

### Task 3 Deliverables

**Due**: Final Submission - Tuesday, 10 Feb 2026, 8:00 PM UTC

Checklist:
- [ ] Flask backend (app.py with all endpoints)
- [ ] React frontend components (Dashboard, Charts, etc.)
- [ ] requirements.txt for backend
- [ ] package.json for frontend
- [ ] Setup guide with installation instructions
- [ ] API documentation
- [ ] Screenshots demonstrating functionality
- [ ] README with deployment instructions
- [ ] Complete Jupyter notebook (JSON format) with Task 3 code
- [ ] Testing code examples

---

## CODE MODULARIZATION PRINCIPLES

### 1. Class-Based Organization

**Principle**: Group related functionality into classes

```python
class DataLoader:
    """Handles all data loading operations"""
    
    def load_data(self):
        """Load data from source"""
    
    def preprocess_data(self):
        """Clean and prepare data"""
    
    def validate_data(self):
        """Validate data quality\"\""

class TimeSeriesAnalyzer:
    """Handles time series analysis"""
    
    def calculate_returns(self):
        """Calculate returns"""
    
    def plot_price_series(self):
        """Create visualizations"""

class BayesianChangePointModel:
    """Handles Bayesian modeling"""
    
    def build_model(self):
        """Build PyMC model"""
    
    def fit_model(self):
        """Run MCMC sampling"""
```

### 2. Separation of Concerns

**Principle**: Each class has single responsibility

- **DataLoader**: Only data loading/validation
- **Analyzer**: Only analysis operations
- **Visualizer**: Only plotting/visualization
- **Interpreter**: Only results interpretation
- **API**: Only request/response handling

### 3. Composition Over Inheritance

```python
# Good: Composition
class Dashboard:
    def __init__(self, data_manager, analyzer, visualizer):
        self.data_manager = data_manager
        self.analyzer = analyzer
        self.visualizer = visualizer

# Avoid: Deep inheritance hierarchies
class AnalysisBase:
    pass

class TimeSeriesAnalysis(AnalysisBase):
    pass

class ChangePointAnalysis(TimeSeriesAnalysis):
    pass
```

### 4. Dependency Injection

```python
# Good: Dependencies passed in
class Analyzer:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

# Avoid: Global dependencies
class Analyzer:
    def __init__(self):
        self.logger = global_logger  # Bad
```

### 5. Configuration Over Code

```python
# Good: Configuration
CONFIG = {
    'draws': 2000,
    'tune': 1000,
    'cores': 2,
    'chains': 2
}

model.fit_model(**CONFIG)

# Avoid: Hard-coded values
model.fit_model(draws=2000, tune=1000, cores=2, chains=2)
```

---

## LOGGING STRATEGY

### 1. Logging Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working
- **WARNING**: Something unexpected happened (default for missing data)
- **ERROR**: A serious problem occurred
- **CRITICAL**: System cannot continue

### 2. Logging Configuration

```python
import logging
import sys

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler('analysis.log')  # File output
    ]
)

logger = logging.getLogger(__name__)
```

### 3. Logging Points by Operation

**Data Loading:**
```python
logger.info('Loading data from BrentOilPrices.csv')
logger.info(f'Data loaded: {len(df)} records')
logger.warning(f'Found {missing_count} missing values')
logger.info(f'Date range: {df[\"Date\"].min()} to {df[\"Date\"].max()}')
```

**EDA:**
```python
logger.info('Starting Exploratory Data Analysis')
logger.info(f'Mean price: ${df[\"Price\"].mean():.2f}')
logger.info(f'Creating visualizations')
logger.info('EDA completed')
```

**Modeling:**
```python
logger.info('Building Bayesian change point model')
logger.info(f'Model structure created with {n} observations')
logger.info('Starting MCMC sampling...')
logger.info('MCMC sampling completed')
logger.info(f'Sampling took {elapsed_time} seconds')
```

**Results:**
```python
logger.info(f'Change point detected at index {tau_median}')
logger.info(f'Mean before: ${mu1_mean:.2f}, Mean after: ${mu2_mean:.2f}')
logger.info('Results interpreted successfully')
```

### 4. Log File Management

```python
# Create task-specific log files
logging.FileHandler('task_1_analysis.log')
logging.FileHandler('task_2_modeling.log')
logging.FileHandler('task_3_dashboard.log')

# Keep logs organized with clear headers
logger.info('='*80)
logger.info('TASK 1: LAYING THE FOUNDATION')
logger.info('='*80)
```

---

## FILE DELIVERABLES

### Task 1 Files
- `Task_1_Foundation.ipynb` (Jupyter notebook as JSON)
- `major_events.csv` (15+ events with structured data)
- `assumptions_and_limitations.txt` (Documentation)
- `01_brent_price_series.png` (Visualization)
- `02_log_returns_volatility.png` (Visualization)
- `task_1_analysis.log` (Logging output)

### Task 2 Files
- `Task_2_ChangePointModeling.ipynb` (Jupyter notebook as JSON)
- `task2_01_eda_visualizations.png`
- `task2_02_rolling_statistics.png`
- `task2_03_trace_plots.png`
- `task2_04_change_point_posterior.png`
- `task2_05_parameter_posteriors.png`
- `model_results.pkl` (Saved model and results)
- `task_2_modeling.log` (Logging output)

### Task 3 Files
- `Task_3_Dashboard.ipynb` (Jupyter notebook as JSON)
- `backend/app.py` (Flask application)
- `backend/requirements.txt` (Python dependencies)
- `frontend/src/components/Dashboard.jsx`
- `frontend/package.json` (JavaScript dependencies)
- `SETUP_GUIDE.md` (Detailed setup instructions)
- `README.md` (Project overview and usage)
- `API_DOCUMENTATION.md` (Endpoint specifications)
- `task_3_dashboard.log` (Logging output)

---

## SUBMISSION CHECKLIST

### Interim Submission (Sunday, 08 Feb 2026)
- [ ] GitHub link to main branch
- [ ] Task 1 Jupyter notebook (JSON)
- [ ] major_events.csv file
- [ ] Assumptions and limitations document (1-2 pages)
- [ ] Task 1 visualizations (3+ PNG files)
- [ ] Initial EDA findings

### Final Submission (Tuesday, 10 Feb 2026)
- [ ] GitHub link with complete code
- [ ] All 3 Jupyter notebooks (JSON format)
- [ ] Final report (blog post format or PDF)
- [ ] All visualizations from all tasks
- [ ] Dashboard screenshots and usage guide
- [ ] Complete API documentation
- [ ] Setup and deployment guide
- [ ] Limitations and future work discussion
- [ ] All log files documenting execution

---

## SUCCESS CRITERIA

**Excellent Work:**
- Deep, well-explained analysis
- Proper use of Bayesian inference
- Comprehensive modularization
- Extensive logging throughout
- Clear communication of results
- Production-ready code
- Professional documentation
- Honest discussion of limitations

**Good Work:**
- Core analysis completed
- Most modularization applied
- Logging at key points
- Results clearly presented
- Working dashboard
- Basic documentation

**Acceptable Work:**
- Analysis completed
- Some code organization
- Minimal logging
- Results documented
- Functional dashboard

---

## RESOURCES

### Documentation
- Scipy Stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- PyMC: https://docs.pymc.io/
- ArviZ: https://arviz-devs.github.io/arviz/
- Recharts: https://recharts.org/
- Flask: https://flask.palletsprojects.com/
- React: https://react.dev/

### Learning Materials
- Bayesian Statistics: https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/steel/steel_homepage/bayesiantsrev.pdf
- Change Point Detection: https://forecastegy.com/posts/change-point-detection-time-series-python/
- MCMC Explained: https://towardsdatascience.com/monte-carlo-markov-chain-mcmc-explained-94e3a6c8de11

---

**Good luck with your analysis! Remember to focus on clear communication, proper documentation, and honest interpretation of your results.**

Generated: 04 Feb 2026  
Version: 1.0  
Organization: Birhan Energies Analysis Team
