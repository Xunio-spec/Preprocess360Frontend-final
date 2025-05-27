import React, { useState, useEffect } from "react";

const AnalyzeHeadersGrid = ({ onHeaderSelect }) => {
  const [headers, setHeaders] = useState([]);

  useEffect(() => {
    // Fetch headers when component mounts
    fetchHeaders();
  }, []);

  const fetchHeaders = async () => {
    try {
      const response = await fetch("http://localhost:5001/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch headers");
      }

      const data = await response.json();
      console.log("Headers fetched:", data);
      setHeaders(data);
    } catch (error) {
      console.error("Error fetching headers:", error);
    }
  };

  return (
    <div className="p-4">
      {headers.length === 0 ? (
        <p>Loading columns... If this persists, please upload a CSV file first.</p>
      ) : (
        <div className="grid grid-cols-4 gap-4">
          {headers.map((header, index) => (
            <label key={index} className="flex items-center space-x-2 ml-2 mr-2">
              <input 
                type="checkbox" 
                value={header} 
                name="options" 
                id="headerCheckBox"
                onChange={() => onHeaderSelect && onHeaderSelect(header)}
              />
              <span>{header}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
};

export default AnalyzeHeadersGrid;

