// Date Range Filter Component
import React from 'react';

const DateRangeFilter = ({ onApply }) => {
  const [startDate, setStartDate] = React.useState('');
  const [endDate, setEndDate] = React.useState('');

  const handleApply = () => {
    onApply({ start: startDate, end: endDate });
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow mb-4">
      <h3 className="font-semibold mb-3">Filter by Date Range</h3>
      <div className="flex gap-4">
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          className="border rounded px-3 py-2"
          placeholder="Start Date"
        />
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          className="border rounded px-3 py-2"
          placeholder="End Date"
        />
        <button
          onClick={handleApply}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
        >
          Apply
        </button>
      </div>
    </div>
  );
};

export default DateRangeFilter;