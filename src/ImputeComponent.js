import React, { useState } from "react";
import AnalyzeHeadersGrid from "./AnalyzeHeaders"; // Import your component

const ImputeComponent = ({ isDialogOpen, selectedNodeData, closeDialog }) => {
  const [mval, setMval] = useState("");
  const [strategy, setStrategy] = useState("mean"); // Default imputation strategy
  const [selectedColumns, setSelectedColumns] = useState([]);

  const handleCheckboxChange = (header) => {
    setSelectedColumns((prevColumns) => {
      const isChecked = prevColumns.includes(header);
      return isChecked
        ? prevColumns.filter((col) => col !== header)
        : [...prevColumns, header];
    });
  };

  const handleImpute = async () => {
    try {
      const response = await fetch("http://localhost:5001/impute", { // Updated port
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded", // Flask-compatible format
        },
        body: new URLSearchParams({
          mval: mval,
          strategy: strategy,
          selectedColumns: JSON.stringify(selectedColumns), // Convert array to string
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to submit data");
      }

      const result = await response.json();
      console.log("Impute response:", result);
      alert("Imputation completed successfully!");
    } catch (error) {
      console.error("Error performing imputation:", error);
      alert("Failed to perform imputation.");
    }
  };

  return (
    isDialogOpen &&
    selectedNodeData.data.label === "Impute" && (
      <div
        className="rounded-lg fixed top-1/2 left-1/2 bg-white shadow-lg border p-4 flex flex-col justify-center items-center transform -translate-x-1/2 -translate-y-1/2 z-50"
      >
        <div className="flex flex-col items-center space-y-4">
          <label htmlFor="mval" className="font-medium">
            What is the missing value you want to impute?
          </label>
          <input
            placeholder="Enter value here"
            type="text"
            name="mval"
            id="mval"
            value={mval}
            onChange={(e) => setMval(e.target.value)}
            className="mb-2 border border-gray-300 rounded px-2 py-1"
          />

          <label htmlFor="strategy" className="font-medium">
            Select Imputation Strategy
          </label>
          <select
            name="strategy"
            id="strategy"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="mb-2 border border-gray-300 rounded px-2 py-1"
          >
            <option value="mean">Mean</option>
            <option value="median">Median</option>
            <option value="most_frequent">Most Frequent</option>
            <option value="constant">Constant</option>
          </select>

          <div className="flex flex-col items-center justify-center mb-4">
            <AnalyzeHeadersGrid
              onHeaderSelect={handleCheckboxChange} // Pass a callback to handle header selection
            />
          </div>

          <button
            onClick={handleImpute}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Impute
          </button>
        </div>

        <button
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 mt-2"
          onClick={closeDialog}
        >
          Close
        </button>
      </div>
    )
  );
};

export default ImputeComponent;

