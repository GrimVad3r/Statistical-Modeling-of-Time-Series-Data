╔════════════════════════════════════════════════════════════════════════════╗
║     COMPLETE SETUP GUIDE FOR BRENT OIL PRICES DASHBOARD                    ║
║     Change Point Analysis - Interactive Dashboard                           ║
╚════════════════════════════════════════════════════════════════════════════╝

## PART 1: ENVIRONMENT SETUP

### 1.1 Backend Setup (Flask)

Step 1: Create backend directory structure
```
mkdir brent-dashboard
cd brent-dashboard
mkdir backend frontend data logs
```

Step 2: Set up Python virtual environment
```
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

Step 3: Install Python dependencies
```
pip install flask
pip install flask-cors
pip install pandas numpy
pip install pymc arviz
pip install gunicorn  # For production
```

Step 4: Create requirements.txt
```
flask==2.3.0
flask-cors==4.0.0
pandas==1.5.0
numpy==1.24.0
pymc==5.0.0
arviz==0.15.0
gunicorn==20.1.0
```

Step 5: Prepare data files in backend/data/
```
cp ../BrentOilPrices.csv data/
cp ../major_events.csv data/
# Also include model_results.pkl from Task 2
```

### 1.2 Frontend Setup (React)

Step 1: Create React application
```
cd frontend
npx create-react-app .
npm install
```

Step 2: Install required packages
```
npm install recharts axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Step 3: Configure Tailwind CSS
```
# Edit tailwind.config.js
content: [
  "./src/**/*.{js,jsx,ts,tsx}",
]
```

## PART 2: FILE STRUCTURE

```
brent-dashboard/
├── backend/
│   ├── venv/
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt        # Python dependencies
│   ├── data/
│   │   ├── BrentOilPrices.csv
│   │   ├── major_events.csv
│   │   └── model_results.pkl
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── prices.py          # Price endpoints
│   │   ├── changepoint.py     # Change point endpoints
│   │   └── events.py          # Events endpoints
│   └── utils/
│       ├── __init__.py
│       └── data_manager.py    # Data loading utilities
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── DateRangeFilter.jsx
│   │   │   ├── PriceChart.jsx
│   │   │   ├── ChangePointCard.jsx
│   │   │   └── EventTimeline.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Analysis.jsx
│   │   │   └── About.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.js
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## PART 3: API ENDPOINT DOCUMENTATION

### Health Check
```
GET /api/v1/health
Response: {"status": "healthy", "timestamp": "...", "version": "1.0.0"}
```

### Historical Prices
```
GET /api/v1/prices?start_date=2020-01-01&end_date=2022-12-31
Response: {
  "data": [{"Date": "2020-01-01", "Price": 65.42}, ...],
  "count": 750,
  "date_range": {"start": "...", "end": "..."}
}
```

### Change Point Analysis
```
GET /api/v1/change-point
Response: {
  "detected_date": "2008-09-15",
  "mean_before": 75.43,
  "mean_after": 55.21,
  "absolute_change": -20.22,
  "percent_change": -26.8,
  "credible_interval": {"lower": 8400, "upper": 8500},
  "volatility": 8.32
}
```

### Major Events
```
GET /api/v1/events
Response: {
  "data": [
    {
      "date": "1990-08-02",
      "event_name": "Iraq Invasion of Kuwait",
      "category": "Geopolitical Conflict",
      "description": "..."
    }, ...
  ],
  "count": 15
}
```

### Statistics
```
GET /api/v1/statistics
Response: {
  "mean": 65.21,
  "median": 63.45,
  "std_dev": 28.34,
  "min": 10.25,
  "max": 147.50,
  "range": 137.25,
  "records": 9000
}
```

## PART 4: RUNNING THE APPLICATION

### Development Mode

#### Terminal 1: Start Flask Backend
```
cd backend
source venv/bin/activate
python app.py
# Output: Running on http://localhost:5000
```

#### Terminal 2: Start React Frontend
```
cd frontend
npm start
# Output: Compiled successfully! App running on http://localhost:3000
```

#### Access Application
```
Open browser: http://localhost:3000
```

### Production Deployment

Step 1: Build React application
```
cd frontend
npm run build
# Creates optimized build in frontend/build/
```

Step 2: Serve static files from Flask
```
# In Flask app.py, add:
from flask import send_from_directory

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')
```

Step 3: Run with Gunicorn
```
cd backend
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

Step 4: Deploy to cloud (AWS, Heroku, GCP)
```
# Example for Heroku:
heroku login
heroku create brent-oil-dashboard
git push heroku main
```

## PART 5: KEY FEATURES IMPLEMENTED

✓ Historical price visualization with interactive chart
✓ Change point detection results display
✓ Event timeline with major geopolitical/economic events
✓ Date range filtering
✓ Summary statistics dashboard
✓ Responsive design (desktop, tablet, mobile)
✓ Error handling and loading states
✓ API documentation
✓ Logging for debugging
✓ CORS support for cross-origin requests

## PART 6: TESTING

### Test Backend API
```
curl http://localhost:5000/api/v1/health
curl http://localhost:5000/api/v1/prices
curl http://localhost:5000/api/v1/change-point
curl http://localhost:5000/api/v1/events
curl http://localhost:5000/api/v1/statistics
```

### Test Frontend Components
```
npm test
# Run React testing library tests
```

## PART 7: TROUBLESHOOTING

Issue: CORS errors
Solution: Ensure flask-cors is installed and CORS(app) is called

Issue: Data not loading
Solution: Verify data files exist in backend/data/ directory

Issue: Port already in use
Solution: Change port in app.py or frontend package.json

Issue: Module not found
Solution: Ensure all dependencies installed via requirements.txt

## PART 8: MONITORING AND LOGS

```
# Backend logs
tail -f backend/logs/app.log

# Frontend browser console
F12 → Console tab

# Check API responses
open DevTools → Network tab → Check requests
```

═════════════════════════════════════════════════════════════════════════════════
For additional support, refer to:
- Flask Documentation: https://flask.palletsprojects.com
- React Documentation: https://react.dev
- Recharts Documentation: https://recharts.org
═════════════════════════════════════════════════════════════════════════════════