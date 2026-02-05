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
class EventCompiler:
    """
    Module for compiling major geopolitical and economic events
    that may have impacted Brent oil prices.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.events = []
    
    def compile_events(self):
        """
        Compile a comprehensive list of major events from 1987-2022.
        Focus on: Geopolitical, OPEC decisions, sanctions, and economic shocks.
        """
        self.events = [
            {
                'date': '1990-08-02',
                'event_name': 'Iraq Invasion of Kuwait',
                'category': 'Geopolitical Conflict',
                'description': 'Iraqi invasion triggers major oil supply concerns and price spike'
            },
            {
                'date': '1991-01-17',
                'event_name': 'Gulf War Begins',
                'category': 'Armed Conflict',
                'description': 'Operation Desert Storm commences, oil markets volatile'
            },
            {
                'date': '1997-07-02',
                'event_name': 'Asian Financial Crisis',
                'category': 'Economic Crisis',
                'description': 'Currency collapse spreads across Asia, demand shock for commodities'
            },
            {
                'date': '2001-09-11',
                'event_name': 'September 11 Attacks',
                'category': 'Terrorism/Geopolitical',
                'description': 'Terrorist attacks in US impact global markets and economic outlook'
            },
            {
                'date': '2003-03-20',
                'event_name': 'Iraq War Begins',
                'category': 'Armed Conflict',
                'description': 'US-led invasion of Iraq, supply disruptions expected'
            },
            {
                'date': '2004-01-01',
                'event_name': 'Oil Prices Begin Sustained Rise',
                'category': 'Market Trend',
                'description': 'Start of the 2004-2008 oil boom, driven by supply constraints and demand'
            },
            {
                'date': '2008-07-11',
                'event_name': 'Oil Prices Peak',
                'category': 'Price Peak',
                'description': 'Brent crude reaches all-time high of ~$145/barrel'
            },
            {
                'date': '2008-09-15',
                'event_name': 'Lehman Brothers Collapse',
                'category': 'Financial Crisis',
                'description': 'Global financial crisis triggers demand collapse and oil price crash'
            },
            {
                'date': '2011-03-15',
                'event_name': 'Libyan Civil War',
                'category': 'Geopolitical Conflict',
                'description': 'Uprising in Libya disrupts major oil production, prices spike'
            },
            {
                'date': '2014-06-01',
                'event_name': 'Oil Price Decline Begins',
                'category': 'Market Trend',
                'description': 'Saudi Arabia increases production, leading to sustained price decline'
            },
            {
                'date': '2014-11-27',
                'event_name': 'OPEC Abandons Production Cuts',
                'category': 'OPEC Policy',
                'description': 'OPEC decision to maintain production accelerates oil price collapse'
            },
            {
                'date': '2016-02-11',
                'event_name': 'Oil Prices Hit Bottom',
                'category': 'Price Low',
                'description': 'Brent crude falls to ~$26/barrel during global supply glut'
            },
            {
                'date': '2016-11-30',
                'event_name': 'OPEC Announces Production Cuts',
                'category': 'OPEC Policy',
                'description': 'OPEC agrees to reduce production to support prices'
            },
            {
                'date': '2020-02-01',
                'event_name': 'COVID-19 Pandemic Begins',
                'category': 'Health Crisis',
                'description': 'Global pandemic triggers economic shutdown and oil demand collapse'
            },
            {
                'date': '2020-04-20',
                'event_name': 'Oil Prices Turn Negative',
                'category': 'Price Anomaly',
                'description': 'May crude futures turn negative for first time in history'
            }
        ]
        
        self.logger.info(f'Compiled {len(self.events)} major events')
        return self.events
    
    def export_to_csv(self, filepath='major_events.csv'):
        """
        Export compiled events to CSV for reference.
        """
        try:
            events_df = pd.DataFrame(self.events)
            events_df.to_csv(filepath, index=False)
            self.logger.info(f'Events exported to {filepath}')
            return events_df
        except Exception as e:
            self.logger.error(f'Error exporting events: {str(e)}')
            raise
    
    def display_events(self):
        """
        Display events in a formatted table.
        """
        events_df = pd.DataFrame(self.events)
        print('\nMAJOR EVENTS IMPACTING BRENT OIL PRICES (1987-2022)')
        print('=' * 100)
        print(events_df.to_string(index=False))
        print('=' * 100)
        return events_df