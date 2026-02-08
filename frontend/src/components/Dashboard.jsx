// src/components/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ReferenceLine 
} from 'recharts';
import axios from 'axios';

const Dashboard = () => {
  const [priceData, setPriceData] = useState([]);
  const [changePoint, setChangePoint] = useState(null);
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  // const [error, setError] = useState(null);   // optional: add later

  useEffect(() => {
  console.log("priceData actually changed in render - length:", priceData.length);
  if (priceData.length > 0) {
    console.log("First real point in state:", priceData[0]);
    console.log("Last real point in state:", priceData[priceData.length - 1]);
    console.log("Min Date:", Math.min(...priceData.map(d => d.Date)));
    console.log("Max Date:", Math.max(...priceData.map(d => d.Date)));
  }
}, [priceData]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);

        // Optional: health check
        await axios.get('http://localhost:5000/api/v1/health');

        // Fetch prices
        const pricesRes = await axios.get('http://localhost:5000/api/v1/prices');
        console.log("Raw prices response:", pricesRes.data);

        const rawItems = Array.isArray(pricesRes.data.data) ? pricesRes.data.data : [];
        console.log("Raw items count:", rawItems.length, "first item:", rawItems[0] || "empty");

        const formatted = rawItems
          .map((item, index) => {
            const dateStr = item.Date ?? item.date ?? null;
            const parsed = dateStr ? new Date(dateStr) : null;
            const ts = parsed && !isNaN(parsed.getTime()) ? parsed.getTime() : null;

            const priceVal = Number(item.Price ?? item.price ?? NaN);

            if (ts === null || isNaN(priceVal)) {
              console.warn(`Invalid row at index ${index}:`, { dateStr, price: item.Price });
              return null;
            }

            return { Date: ts, Price: priceVal };
          })
          .filter(Boolean);   // remove nulls

        console.log(
          "Formatted count:", formatted.length,
          "Sample (first 2):", formatted.slice(0, 2),
          "Sample (last 2):", formatted.slice(-2)
        );

        setPriceData(formatted);

        // Log AFTER set (will show old value - normal)
        console.log("Immediately after setPriceData - current priceData length:", priceData.length);

        // Fetch change point
        const cpRes = await axios.get('http://localhost:5000/api/v1/change-point');
        setChangePoint(cpRes.data);
        console.log("Change point set:", cpRes.data);             // should log the full object

        // Fetch events
        const eventsRes = await axios.get('http://localhost:5000/api/v1/events');
        setEvents(eventsRes.data.data);
        console.log("Events set:", eventsRes.data.data?.length);  // number of events

        // Fetch statistics
        const statsRes = await axios.get('http://localhost:5000/api/v1/statistics');
        setStats(statsRes.data);
        console.log("Stats set:", statsRes.data);                 // full stats object

      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        // setError(error.message);   // optional
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);  // empty dependency array → run once on mount

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p className="text-xl font-medium">Loading Brent Oil Dashboard...</p>
      </div>
    );
  }

  return (
  <div className="min-h-screen bg-gray-50 py-8">
      {/* No max-w, no container – full bleed */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-10">
          Brent Oil Prices – Change Point Analysis
        </h1>
      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <div className="bg-white p-6 rounded-xl shadow-md">
          <p className="text-sm text-gray-600 font-medium">Mean Price</p>
          <p className="text-3xl font-bold text-blue-700 mt-1">
            ${stats?.mean?.toFixed(2) ?? '—'}
          </p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-md">
          <p className="text-sm text-gray-600 font-medium">Max Price</p>
          <p className="text-3xl font-bold text-green-700 mt-1">
            ${stats?.max?.toFixed(2) ?? '—'}
          </p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-md">
          <p className="text-sm text-gray-600 font-medium">Min Price</p>
          <p className="text-3xl font-bold text-red-700 mt-1">
            ${stats?.min?.toFixed(2) ?? '—'}
          </p>
        </div>
        <div className="bg-white p-6 rounded-xl shadow-md">
          <p className="text-sm text-gray-600 font-medium">Volatility (Std Dev)</p>
          <p className="text-3xl font-bold text-purple-700 mt-1">
            ${stats?.std_dev?.toFixed(2) ?? '—'}
          </p>
        </div>
      </div>

      {/* Chart – generous height & full width */}
      <div className="bg-white rounded-xl shadow-lg mb-12 overflow-hidden">
        <div className="p-6 lg:p-10"></div>
          <h2 className="text-3xl font-bold mb-8">Historical Price Trends</h2>
          <div className="w-full h-[500px]"> {/* 70% of viewport height */}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={priceData}
              margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="Date"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(ts) => new Date(ts).toLocaleDateString('en-US', { year: 'numeric', month: 'short' })}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis tickFormatter={(v) => `$${Math.round(v)}`} />
              <Tooltip
                labelFormatter={(ts) => new Date(ts).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                formatter={(v) => [`$${v.toFixed(2)}`, 'Price']}
              />
              <Legend verticalAlign="top" height={36} />
              <Line
                type="monotone"
                dataKey="Price"
                stroke="#2563eb"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 8 }}
              />
              {changePoint && (
                <ReferenceLine
                  x={new Date(changePoint.detected_date).getTime()}
                  stroke="#ef4444"
                  strokeDasharray="6 6"
                  strokeWidth={2}
                  label={{
                    value: `Change Point (${new Date(changePoint.detected_date).toLocaleDateString()})`,
                    position: 'insideTop',
                    fill: '#ef4444',
                    fontSize: 13
                  }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Change Point Details – add your full grid here */}
      {changePoint && (
        <div className="bg-white p-6 md:p-8 rounded-xl shadow-md">
          <h2 className="text-2xl md:text-3xl font-bold mb-6">Change Point Analysis</h2>
          {/* your grid with Detected Date, Mean Before, Mean After, etc. */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <p className="text-gray-600 font-medium">Detected Date</p>
              <p className="text-2xl font-semibold text-red-600 mt-1">
                {new Date(changePoint.detected_date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </p>
            </div>
            {/* ... other metrics ... */}
          </div>
        </div>
      )}

      {/* Events – if implemented */}
    </div>
  </div>
);
};

export default Dashboard;