import React, { useState } from 'react';

const DropdownMenu = () => {
  // State to track selected option
  const [selectedOption, setSelectedOption] = useState('');

  // Handle dropdown selection
  const handleSelect = (event) => {
    setSelectedOption(event.target.value);
  };

  return (
    <div>
      <h3>Choose an Option:</h3>
      
      {/* Dropdown menu */}
      <select id='strategyDrop' value={selectedOption} onChange={handleSelect}>
        <option value="">-- Select --</option>
        <option value="mean">Mean</option>
        <option value="median">Median</option>
        <option value="most_frequent">Most Frequent</option>
        
      </select>

      {/* Display selected option */}
      {selectedOption && <p>You selected: {selectedOption}</p>}
    </div>
  );
};

export default DropdownMenu;
