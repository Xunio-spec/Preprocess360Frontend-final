import React, { useState, useEffect } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  Handle,
} from 'reactflow';
import 'reactflow/dist/style.css';
import DropdownMenu from './DropdownMenu';
import AnalyzeHeadersGrid from './AnalyzeHeaders';
import ImputeComponent from './ImputeComponent';
import { Database, Table, ChartBar, Trash, Moon, Sun, Brain } from 'lucide-react';

// Node Components
const NodeWrapper = ({ children, label }) => (
  <div className="flex flex-col items-center">
    {children}
    <span className="text-xs font-medium mt-2">{label}</span>
  </div>
);

const DataNode = ({ data }) => (
  <NodeWrapper label={data.label}>
    <div
      className="bg-white p-3 rounded-full shadow-sm w-10 h-10 flex items-center justify-center border-2 border-zinc-300"
      style={{
        background: 'radial-gradient(circle farthest-side, #FEDEB6 50%, #FEC27B)',
      }}
    >
      <Handle type="target" position="left" style={{ background: '#555' }} />
      <Database className="w-5 h-5" />
      <Handle type="source" position="right" style={{ background: '#555' }} />
    </div>
  </NodeWrapper>
);

const PreprocessNode = ({ data }) => (
  <NodeWrapper label={data.label}>
    <div
      className="bg-white p-3 rounded-full shadow-sm w-10 h-10 flex items-center justify-center border-2 border-zinc-300"
      style={{
        background: 'radial-gradient(circle farthest-side, #FEDEB6 50%, #FEC27B)',
      }}
    >
      <Handle type="target" position="left" style={{ background: '#555' }} />
      <Table className="w-5 h-5" />
      <Handle type="source" position="right" style={{ background: '#555' }} />
    </div>
  </NodeWrapper>
);

const VisualizationNode = ({ data }) => (
  <NodeWrapper label={data.label}>
    <div
      className="bg-white p-3 rounded-full shadow-sm w-10 h-10 flex items-center justify-center border-2 border-zinc-300"
      style={{
        background: 'radial-gradient(circle farthest-side, #FEDEB6 50%, #FEC27B)',
      }}
    >
      <Handle type="target" position="left" style={{ background: '#555' }} />
      <ChartBar className="w-5 h-5" />
      <Handle type="source" position="right" style={{ background: '#555' }} />
    </div>
  </NodeWrapper>
);

const nodeTypes = {
  data: DataNode,
  preprocess: PreprocessNode,
  visualize: VisualizationNode,
  ml: ({ data }) => (
    <NodeWrapper label={data.label}>
      <div
        className="bg-white p-3 rounded-full shadow-sm w-10 h-10 flex items-center justify-center border-2 border-zinc-300"
        style={{
          background: 'radial-gradient(circle farthest-side, #FEDEB6 50%, #FEC27B)',
        }}
      >
        <Handle type="target" position="left" style={{ background: '#555' }} />
        <Brain className="w-5 h-5" />
        <Handle type="source" position="right" style={{ background: '#555' }} />
      </div>
    </NodeWrapper>
  ),
};

const WorkflowCanvas = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([
    {
      id: 'data-1',
      type: 'data',
      data: { label: 'Data Node' },
      position: { x: 100, y: 100 },
    },
  ]);

  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const [visualizationDropdownVisible, setVisualizationDropdownVisible] = useState(false);
  const [dataDropdownVisible, setDataDropdownVisible] = useState(false);  // New state for "Add Data" dropdown
  const [mlDropdownVisible, setMlDropdownVisible] = useState(false);  // New state for "Machine Learning" dropdown
  const [darkMode, setDarkMode] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false); // Tracks dialog visibility
  const [dialogContent, setDialogContent] = useState('');  // Stores dialog content
  const [data, setData] = useState(null);
  const [file, setFile] = useState(null);
  const [selectedNodeData, setSelectedNodeData] = useState(null);
  const [showPlotModal, setShowPlotModal] = useState(false);
  const [currentPlot, setCurrentPlot] = useState(null);
  const [plotUrl, setPlotUrl] = useState('');
  const [mlResults, setMlResults] = useState(null);
  const [showMlResultsModal, setShowMlResultsModal] = useState(false);
  const [headers, setHeaders] = useState([]);

  useEffect(() => {
    // Fetch headers from the server
    const fetchHeaders = async () => {
      try {
        const response = await fetch('http://localhost:5001/headers');
        if (response.ok) {
          const data = await response.json();
          setHeaders(data.headers || []);
        }
      } catch (error) {
        console.error('Error fetching headers:', error);
      }
    };
    
    fetchHeaders();
  }, []);

  const handleFileUpload = (e) => {
    const selectedFile = e.target.files[0];
    console.log("Selected file:", selectedFile);
    setFile(selectedFile);
  };

  const uploadFile = async () => {
    console.log("Starting file upload, file:", file);
    if (!file) return alert('Please select a file.');  // Check if file is selected
    
    const formData = new FormData();
    formData.append('file', file);  // Append the file to the FormData object
    console.log("FormData created with file:", file.name);

    try {
      console.log("Sending request to server...");
      // Send the formData using fetch
      const response = await fetch('http://127.0.0.1:5001/upload', {  // Updated port
        method: 'POST',  // Use POST method
        body: formData,  // Send formData as the request body
      });

      console.log("Response received:", response.status);
      // Check if the response is okay (status code 200-299)
      if (response.ok) {
        const data = await response.json();  // Parse the JSON response
        console.log("Response data:", data);
        setData(data);
        alert("File uploaded successfully!");
        
        // Call analyzeFile after successful upload
        await analyzeFile();
      } else {
        const errorText = await response.text();
        console.error("Upload failed:", errorText);
        throw new Error(`File upload failed: ${errorText}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);  // Log any errors
      alert(`Error uploading file: ${error.message}`);
    }
  };



  const analyzeFile = async () => {
    try {
      const response = await fetch('http://localhost:5001/analyze', { // Updated port
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze file');
      }

      const headers = await response.json();
      console.log(headers);

      // Display the headers in the UI
      const outputElement = document.getElementById('arrayOutput');
      if (outputElement) {
        outputElement.textContent = `Headers: ${headers.join(', ')}`;
      }
    } catch (error) {
      console.error('Error analyzing file:', error);
    }
  };



  const onNodeDoubleClick = async (_, node) => {
    // Check the node type or label
    if (node.type === 'data' || node.data.label === 'Impute' || node.data.label === 'Standardize Data' || node.data.label === 'Delete Outliers') {
      setIsDialogOpen(true);          // Open the dialog
      setDialogContent(node.data.label); // Set the dialog content
      setSelectedNodeData(node);      // Store the entire node in state
    }
    // Handle visualization nodes
    else if (node.type === 'visualize') {
      // Set the current plot type and selected node data
      setCurrentPlot(node.data.label);
      setSelectedNodeData(node);
      
      // Open the visualization dialog instead of immediately creating the plot
      setIsDialogOpen(true);
      setDialogContent(node.data.label);
    }
    // Handle ML nodes
    else if (node.type === 'ml') {
      // Fetch headers before opening the dialog
      try {
        const response = await fetch('http://localhost:5001/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),
        });
        
        if (response.ok) {
          const columnHeaders = await response.json();
          // Instead of using setHeaders, store the headers directly in the state
          // using the useState hook that's already defined
          setSelectedNodeData({
            ...node,
            headers: columnHeaders // Store headers in the node data
          });
          console.log("Fetched headers for ML node:", columnHeaders);
        } else {
          console.error("Failed to fetch headers for ML node");
        }
      } catch (error) {
        console.error("Error fetching headers for ML node:", error);
      }
      
      setIsDialogOpen(true);
      setDialogContent(node.data.label);
    }
  };

  // Add a new function to handle visualization with selected columns
  const createVisualizationWithColumns = async () => {
    // Get selected columns
    const checkBoxes = document.getElementsByName('options');
    const selectedColumns = [];
    
    checkBoxes.forEach(box => {
      if (box.checked) {
        selectedColumns.push(box.value);
      }
    });
    
    console.log("Selected columns for visualization:", selectedColumns);
    
    if (selectedColumns.length === 0) {
      alert("Please select at least one column for visualization.");
      return;
    }
    
    // Determine the plot endpoint based on the current plot type
    let plotEndpoint = '';
    if (currentPlot === 'Bar Plot') plotEndpoint = 'barplot';
    else if (currentPlot === 'Line Plot') plotEndpoint = 'lineplot';
    else if (currentPlot === 'Scatter Plot') plotEndpoint = 'scatterplot';
    else if (currentPlot === 'Heat Map') plotEndpoint = 'heatmap';
    else if (currentPlot === 'Violin Plot') plotEndpoint = 'violinplot';
    else if (currentPlot === 'Silhouette Plot') plotEndpoint = 'silhouetteplot';
    
    try {
      console.log(`Creating ${currentPlot} with columns:`, selectedColumns);
      
      const response = await fetch(`http://127.0.0.1:5001/${plotEndpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ columns: selectedColumns }),
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`${currentPlot} result:`, result);
        
        // Set the plot URL with a timestamp to prevent caching
        const timestamp = new Date().getTime();
        const plotUrl = `http://127.0.0.1:5001/static/${plotEndpoint}.png?t=${timestamp}`;
        setPlotUrl(plotUrl);
        
        // Close the dialog and show the plot modal
        setIsDialogOpen(false);
        setShowPlotModal(true);
        
        console.log(`${currentPlot} created successfully!`);
      } else {
        const errorText = await response.text();
        console.error(`Error creating ${currentPlot}:`, errorText);
        alert(`Error: Could not create ${currentPlot}. Server responded with: ${response.status} ${errorText}`);
      }
    } catch (error) {
      console.error(`Error creating ${currentPlot}:`, error);
      alert(`Error: Could not create ${currentPlot}. ${error.message}`);
    }
  };

  // Function to create plots with improved error handling
  const createPlot = async (plotType, plotLabel) => {
    try {
      console.log(`Creating ${plotLabel}...`);
      console.log(`Sending request to: http://127.0.0.1:5001/${plotType}`);
      
      const response = await fetch(`http://127.0.0.1:5001/${plotType}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Adding a timeout to prevent hanging requests
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });

      console.log(`Response status for ${plotLabel}: ${response.status}`);
      
      if (response.ok) {
        const result = await response.json();
        console.log(`${plotLabel} result:`, result);
        
        // Set the plot URL with a timestamp to prevent caching
        const timestamp = new Date().getTime();
        const plotUrl = `http://127.0.0.1:5001/static/${plotType}.png?t=${timestamp}`;
        setPlotUrl(plotUrl);
        
        // Show the plot modal
        setShowPlotModal(true);
        
        console.log(`${plotLabel} created successfully!`);
      } else {
        const errorText = await response.text();
        console.error(`Error creating ${plotLabel}:`, errorText);
        alert(`Error: Could not create ${plotLabel}. Server responded with: ${response.status} ${errorText}`);
      }
    } catch (error) {
      console.error(`Error creating ${plotLabel}:`, error);
      
      // More specific error message
      if (error.name === 'AbortError') {
        alert(`Request timeout: The server took too long to respond. Please check if the server is running.`);
      } else if (error.message.includes('Failed to fetch')) {
        alert(`Network error: Could not connect to the server at http://127.0.0.1:5001. Please check if the server is running.`);
      } else {
        alert(`Error: Could not create ${plotLabel}. ${error.message}`);
      }
    }
  };

  const closeDialog = () => {
    setIsDialogOpen(false);          // Close the dialog
    setDialogContent('');            // Clear dialog content
    setSelectedNodeData(null);       // Clear the selected node data
  };


  // const onConnect = async (params) => {
  //   setEdges((eds) => addEdge(params, eds)); // Add the edge to the canvas

  //   // Check if the connection is between "Data Node" and "Remove Duplicates Node"
  //   console.log(params);
  //   if (params.source.includes('data') && params.target.includes('preprocess')) {
  //     try {
  //       // Call the Flask endpoint to remove duplicates
  //       const response = await fetch('http://127.0.0.1:5000/removeduplicates', {
  //         method: 'POST',
  //       });

  //       if (response.ok) {
  //         const result = await response.json(); // Parse the response JSON
  //         alert(`Duplicates removed: ${result.duplicate_count}`);
  //         console.log(`Duplicates removed: ${result.duplicate_count}`);
  //       } else {
  //         console.error('Failed to remove duplicates.');
  //         alert('Error: Could not remove duplicates.');
  //       }
  //     } catch (error) {
  //       console.error('Error connecting to the backend:', error);
  //       alert('Error connecting to the server.');
  //     }
  //   }
  // };

  const onConnect = async (params) => {
    setEdges((eds) => addEdge(params, eds)); // Add the edge to the canvas

    // Check if the connection is between "Data Node" and the specific "Remove Duplicates Node"
    // console.log(params.target);
    if (params.source.includes('data') && params.target.includes('Duplicates')) {
      try {
        // Call the Flask endpoint to remove duplicates
        const response = await fetch('http://127.0.0.1:5001/removeduplicates', { // Updated port
          method: 'POST',
        });

        if (response.ok) {
          const result = await response.json();
          alert(`Duplicates removed: ${result.duplicate_count}`);
          console.log(`Duplicates removed: ${result.duplicate_count}`);
        } else {
          console.error('Failed to remove duplicates.');
          alert('Error: Could not remove duplicates.');
        }
      } catch (error) {
        console.error('Error connecting to the backend:', error);
        alert('Error connecting to the server.');
      }
    }

    else if ((params.source.includes('Standardize') && params.target.includes('Outliers')) || (params.source.includes('data') && params.target.includes('Outliers'))) {
      try {
        // Call the Flask endpoint to remove outliers
        console.log("Attempting to remove outliers...");
        
        // Make sure we're using the correct URL with http:// prefix
        const response = await fetch('http://127.0.0.1:5001/removeoutlier', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),  // Send an empty JSON object
        });

        console.log("Response status:", response.status);
        
        if (response.ok) {
          const result = await response.json();
          console.log("Outliers removal result:", result);
          alert(`Outliers removed: ${result['Outliers Count']}`);
        } else {
          const errorText = await response.text();
          console.error('Failed to remove outliers:', errorText);
          alert('Error: Could not remove outliers. ' + errorText);
        }
      } catch (error) {
        console.error('Error connecting to the backend:', error);
        alert('Error connecting to the server: ' + error.message);
      }
    }
    else if (params.target.includes('Save Data')) {
      try {
        // Call the Flask endpoint to remove duplicates
        // alert("This shit is working");
        await fetch('http://127.0.0.1:5000/download', {
          method: 'GET',
        })
          .then(response => response.blob())  // Convert the response to a Blob
          .then(blob => {
            // Create a link element, use it to trigger the download, then remove it
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'Preprocess360.csv';  // The file name to be downloaded
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
          })

      } catch (error) {
        console.error('Error connecting to the backend:', error);
        alert('Error connecting to the server.');
      }
    }
    else if ((params.source.includes('data') && params.target.includes('Bar Plot')) || 
             (params.source.includes('Standardize') && params.target.includes('Bar Plot'))) {
      try {
        const response = await fetch('http://127.0.0.1:5000/barplot', {
          method: 'POST',
        });

        if (response.ok) {
          alert('Bar plot created successfully!');
        } else {
          alert('Error: Could not create bar plot.');
        }
      } catch (error) {
        console.error('Error creating bar plot:', error);
        alert('Error: Could not create bar plot.');
      }
    }
    else if ((params.source.includes('data') && params.target.includes('Line Plot')) || 
             (params.source.includes('Standardize') && params.target.includes('Line Plot'))) {
      try {
        const response = await fetch('http://127.0.0.1:5000/lineplot', {
          method: 'POST',
        });

        if (response.ok) {
          alert('Line plot created successfully!');
        } else {
          alert('Error: Could not create line plot.');
        }
      } catch (error) {
        console.error('Error creating line plot:', error);
        alert('Error: Could not create line plot.');
      }
    }
    else if ((params.source.includes('data') && params.target.includes('Scatter Plot')) || 
             (params.source.includes('Standardize') && params.target.includes('Scatter Plot'))) {
      try {
        const response = await fetch('http://127.0.0.1:5000/scatterplot', {
          method: 'POST',
        });

        if (response.ok) {
          alert('Scatter plot created successfully!');
        } else {
          alert('Error: Could not create scatter plot.');
        }
      } catch (error) {
        console.error('Error creating scatter plot:', error);
        alert('Error: Could not create scatter plot.');
      }
    }
    else if ((params.source.includes('data') && params.target.includes('Heat Map')) || 
             (params.source.includes('Standardize') && params.target.includes('Heat Map'))) {
      try {
        const response = await fetch('http://127.0.0.1:5000/heatmap', {
          method: 'POST',
        });

        if (response.ok) {
          alert('Heat map created successfully!');
        } else {
          alert('Error: Could not create heat map.');
        }
      } catch (error) {
        console.error('Error creating heat map:', error);
        alert('Error: Could not create heat map.');
      }
    }
    else if ((params.source.includes('data') && params.target.includes('Violin Plot')) || 
             (params.source.includes('Standardize') && params.target.includes('Violin Plot'))) {
      try {
        const response = await fetch('http://127.0.0.1:5000/violinplot', {
          method: 'POST',
        });

        if (response.ok) {
          alert('Violin plot created successfully!');
        } else {
          alert('Error: Could not create violin plot.');
        }
      } catch (error) {
        console.error('Error creating violin plot:', error);
        alert('Error: Could not create violin plot.');
      }
    }
    else if ((params.source.includes('data') && params.target.includes('Silhouette Plot')) || 
             (params.source.includes('Standardize') && params.target.includes('Silhouette Plot'))) {
      try {
        const response = await fetch('http://127.0.0.1:5000/silhouetteplot', {
          method: 'POST',
        });

        if (response.ok) {
          alert('Silhouette plot created successfully!');
        } else {
          alert('Error: Could not create silhouette plot.');
        }
      } catch (error) {
        console.error('Error creating silhouette plot:', error);
        alert('Error: Could not create silhouette plot.');
      }
    }
  };


  const onNodeClick = (_, node) => {
    setSelectedNode(node);
    setSelectedEdge(null);
    
    // Check if the clicked node is a visualization node
    if (node.type === 'visualize') {
      const plotType = node.data.label.toLowerCase().replace(' ', '');
      
      // Set the current plot type
      setCurrentPlot(node.data.label);
      
      // Determine the plot URL based on the node label
      let url = '';
      if (node.data.label === 'Bar Plot') url = 'http://127.0.0.1:5001/static/barplot.png';
      else if (node.data.label === 'Line Plot') url = 'http://127.0.0.1:5001/static/lineplot.png';
      else if (node.data.label === 'Scatter Plot') url = 'http://127.0.0.1:5001/static/scatterplot.png';
      else if (node.data.label === 'Heat Map') url = 'http://127.0.0.1:5001/static/heatmap.png';
      else if (node.data.label === 'Violin Plot') url = 'http://127.0.0.1:5001/static/violinplot.png';
      else if (node.data.label === 'Silhouette Plot') url = 'http://127.0.0.1:5001/static/silhouetteplot.png';
      
      setPlotUrl(url);
      // We don't show the modal on click anymore, only on double-click
    }
  };

  const onEdgeClick = (_, edge) => {
    setSelectedEdge(edge);
    setSelectedNode(null);
  };

  const deleteSelectedNode = () => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((n) => n.id !== selectedNode.id));
      setEdges((eds) =>
        eds.filter(
          (edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id
        )
      );
      setSelectedNode(null);
    }
  };

  const deleteSelectedEdge = () => {
    if (selectedEdge) {
      setEdges((eds) => eds.filter((edge) => edge.id !== selectedEdge.id));
      setSelectedEdge(null);
    }
  };

  const createNode = (type, position, label) => {
    const newNode = {
      id: `${type}-${label}-${nodes.length + 1}`,
      type,
      data: { label },
      position,
    };
    setNodes((nds) => [...nds, newNode]);
  };

  const toggleDropdown = () => {
    setDropdownVisible((prev) => !prev);
  };

  const toggleVisualizationDropdown = () => {
    setVisualizationDropdownVisible((prev) => !prev);
  };

  const toggleDataDropdown = () => {  // Function to toggle the "Add Data" dropdown
    setDataDropdownVisible((prev) => !prev);
  };

  const toggleMlDropdown = () => {
    setMlDropdownVisible(!mlDropdownVisible);
    // Close other dropdowns when opening this one
    if (!mlDropdownVisible) {
      setDropdownVisible(false);
      setDataDropdownVisible(false);
      setVisualizationDropdownVisible(false);
    }
  };



  const handlePreprocessChange = (action) => {
    setDropdownVisible(false);

    let label = '';
    switch (action) {
      case 'removeDuplicates':
        label = "Remove Duplicates";
        break;
      case 'removeNullValues':
        label = 'Remove Null Values';
        break;
      case 'deleteOutliers':
        label = 'Delete Outliers';
        break;
      case 'normalizeData':
        label = 'Normalize Data';
        break;
      case 'standardizeData':
        label = 'Standardize Data';
        break;
      case 'impute':
        label = 'Impute';
        break;
      case 'unique':
        label = 'Display Unique';
        break;

      default:
        label = 'Preprocess Action';
    }

    createNode('preprocess', { x: 300, y: 100 }, label);
  };

  const handleDataChange = (action) => {
    setDropdownVisible(false); // Close the dropdown after selection

    let label = '';
    switch (action) {
      case 'file':
        label = 'File';
        break;
      case 'csvFileImport':
        label = 'CSV File';
        break;
      case 'datasets':
        label = 'Datasets';
        break;
      case 'sqlTable':
        label = 'SQL Table';
        break;
      case 'dataTable':
        label = 'Data Table';
        break;
      case 'dataInfo':
        label = 'Data Info';
        break;
      case 'saveData':
        label = 'Save Data';
        break;
      default:
        label = 'Data';
    }

    createNode('data', { x: 100, y: 100 }, label); // Create the node with the selected label
  };


  const handleVisualizationChange = (action) => {
    setVisualizationDropdownVisible(false);

    let label = '';
    switch (action) {
      case 'createChart':
        label = 'Create Chart';
        break;
      case 'createGraph':
        label = 'Create Graph';
        break;
      case 'createBar':
        label = 'Bar Plot';
        break;
      case 'createLine':
        label = 'Line Plot';
        break;
      case 'createViolin':
        label = 'Violin Plot';
        break;
      case 'createScatter':
        label = 'Scatter Plot';
        break;
      case 'createSilhouette':
        label = 'Silhouette Plot';
        break;
      case 'createHeatMap':
        label = 'Heat Map';
        break;
      default:
        label = 'Visualization Action';
    }

    createNode('visualize', { x: 500, y: 100 }, label);
  };

  const handleMlChange = (action) => {
    setMlDropdownVisible(false);

    let label = '';
    switch (action) {
      case 'linearRegression':
        label = 'Linear Regression';
        break;
      case 'logisticRegression':
        label = 'Logistic Regression';
        break;
      case 'randomForest':
        label = 'Random Forest';
        break;
      default:
        label = 'ML Algorithm';
    }

    createNode('ml', { x: 300, y: 100 }, label);
  };

  async function dataImputer() {
    // Removed the missingVal variable since we removed the input field
    const checkBoxes = document.getElementsByName('options');
    const strategy = document.getElementById('strategyDrop').value;

    // Collect selected columns
    const columns = [];
    checkBoxes.forEach(box => {
      if (box.checked) {
        columns.push(box.value);
        console.log(box.value + ' is checked');
      }
    });

    console.log("Selected Columns:", columns);
    console.log("Strategy:", strategy);

    // Create form data
    const formData = new FormData();
    // We're not sending mval anymore
    formData.append('columns', JSON.stringify(columns));
    formData.append('strategy', strategy);

    try {
      const response = await fetch('http://localhost:5001/impute', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Impute Result:", result);
      alert("Data imputation completed successfully!");
    } catch (error) {
      console.error("Error during data imputation:", error);
      alert("An error occurred while imputing data.");
    }
  }

  async function standardizeData() {
    const checkBoxes = document.getElementsByName('options');

    // Collect selected columns
    const columns = [];
    checkBoxes.forEach(box => {
      if (box.checked) {
        columns.push(box.value);
        console.log(box.value + ' is checked');
      }
    });

    // Create form data
    const formData = new FormData();
    formData.append('columns', JSON.stringify(columns));

    try {
      const response = await fetch('http://localhost:5001/standardize', { // Updated port
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Standardization Result:", result);
      alert("Selected columns Standardized successfully!");
    } catch (error) {
      console.error("Error during data standardization:", error);
      alert("An error occurred while standardizing data.");
    }
  }

  // const checkConnection = (nodeId1, nodeId2) => {
  //   const isConnected = elements.some(
  //     (edge) => edge.source === nodeId1 && edge.target === nodeId2
  //   );
  //   return isConnected;
  // };

  const testServerConnection = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/test', {
        method: 'GET',
      });
      
      if (response.ok) {
        alert('Connection to server successful!');
      } else {
        alert('Connection failed: ' + response.status);
      }
    } catch (error) {
      alert('Failed to connect: ' + error.message);
    }
  };

  // Add a test button in your UI temporarily
  <button onClick={testServerConnection}>Test Server Connection</button>

  // Add this function to test server connectivity
  const testOutlierEndpoint = async () => {
    try {
      console.log("Testing outlier endpoint...");
      const response = await fetch('http://127.0.0.1:5001/test', {
        method: 'GET',
      });
      
      if (response.ok) {
        console.log("Server connection successful");
        alert('Connection to server successful!');
      } else {
        console.log("Server connection failed:", response.status);
        alert('Connection failed: ' + response.status);
      }
    } catch (error) {
      console.error("Connection error:", error);
      alert('Failed to connect: ' + error.message);
    }
  };

  // Add this function to test server connectivity for visualization endpoints
  const testVisualizationEndpoint = async () => {
    try {
      console.log("Testing visualization endpoint...");
      const response = await fetch('http://127.0.0.1:5001/test', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        console.log("Server connection successful");
        alert('Connection to server successful!');
        return true;
      } else {
        console.log("Server connection failed:", response.status);
        alert('Connection failed: ' + response.status);
        return false;
      }
    } catch (error) {
      console.error("Connection error:", error);
      alert('Failed to connect to server: ' + error.message);
      return false;
    }
  };

  // You can call this function from a button or during initialization
  // testOutlierEndpoint();

  const runMlAlgorithm = async () => {
    const checkBoxes = document.getElementsByName('options');
    const targetColumn = document.getElementById('targetColumn').value;
    const testSize = document.getElementById('testSize').value;
    
    // For Random Forest, check if we need additional parameters
    let taskType = 'classification';
    let nEstimators = 100;
    
    if (dialogContent === 'Random Forest') {
      taskType = document.getElementById('taskType').value;
      nEstimators = document.getElementById('nEstimators').value;
    }
    
    // Collect selected feature columns
    const features = [];
    checkBoxes.forEach(box => {
      if (box.checked && box.value !== targetColumn) {
        features.push(box.value);
      }
    });
    
    if (features.length === 0) {
      alert('Please select at least one feature column');
      return;
    }
    
    if (!targetColumn) {
      alert('Please select a target column');
      return;
    }
    
    // Determine the endpoint based on the algorithm
    let endpoint = '';
    if (dialogContent === 'Linear Regression') endpoint = 'linear-regression';
    else if (dialogContent === 'Logistic Regression') endpoint = 'logistic-regression';
    else if (dialogContent === 'Random Forest') endpoint = 'random-forest';
    
    console.log(`Running ${dialogContent} with endpoint: http://localhost:5001/${endpoint}`);
    console.log("Features:", features);
    console.log("Target:", targetColumn);
    console.log("Test size:", testSize);
    
    try {
      // Add a timeout to the fetch request
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(`http://localhost:5001/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          features,
          target: targetColumn,
          test_size: testSize,
          task_type: taskType,
          n_estimators: nEstimators
        }),
        signal: controller.signal
      });
      
      // Clear the timeout
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} ${errorText}`);
      }
      
      const result = await response.json();
      console.log("ML Result:", result);
      
      // Process the classification_report if it's an object
      if (result.metrics && result.metrics.classification_report && 
          typeof result.metrics.classification_report === 'object') {
        // Convert the classification report object to a formatted string
        result.classification_report = JSON.stringify(result.metrics.classification_report, null, 2);
      }
      
      // Show results in a modal
      setMlResults(result);
      setShowMlResultsModal(true);
      
      // Close the configuration dialog
      setIsDialogOpen(false);
      
    } catch (error) {
      console.error("Error running ML algorithm:", error);
      
      // Provide more specific error messages
      if (error.name === 'AbortError') {
        alert('Request timed out. The server took too long to respond.');
      } else if (error.message.includes('Failed to fetch')) {
        alert(`Network error: Could not connect to the server at http://localhost:5001. Please check if the server is running.`);
      } else {
        alert(`An error occurred: ${error.message}`);
      }
    }
  };

  return (
    <ReactFlowProvider>
      <div
        className={`flex h-screen ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-black'}`}
      >
        <div className="w-1/4 p-4 border-r">
          <h3 className="font-bold text-lg mb-4">Tools</h3>

          {/* Add Data button with dropdown */}
          <button
            className="flex items-center px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-blue-600 mb-2 w-full"
            onClick={toggleDataDropdown} // This will toggle the dropdown visibility
          >
            <Database className="w-4 h-4 mr-2" />
            Data
          </button>

          {dataDropdownVisible && (
            <div className="bg-gray-200 rounded-lg shadow-lg p-2 mt-2 mb-2">
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('file')}
              >
                File
              </button>
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('csvFileImport')}
              >
                CSV File
              </button>
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('datasets')}
              >
                Datasets
              </button>
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('sqlTable')}
              >
                SQL Table
              </button>
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('dataTable')}
              >
                Data Table
              </button>
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('dataInfo')}
              >
                Data Info
              </button>
              <button
                className="block w-full px-4 py-2 text-left"
                onClick={() => handleDataChange('saveData')}
              >
                Save Data
              </button>
            </div>
          )}


          <button
            className="flex items-center px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-blue-600 mb-2 w-full"
            onClick={toggleDropdown}
          >
            <Table className="w-4 h-4 mr-2" />
            Transform
          </button>
          {dropdownVisible && (
            <div className="bg-gray-200 rounded-lg shadow-lg p-2 mt-2 mb-2">
              <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('removeDuplicates')}>
                Remove Duplicate Values
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('removeNullValues')}>
                Remove Null Values
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('deleteOutliers')}>
                Delete Outliers
              </button>
              {/* <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('normalizeData')}>
                Normalize Data
              </button> */}
              <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('standardizeData')}>
                Standardize Data
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('impute')}>
                Impute
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handlePreprocessChange('unique')}>
                Display Unique
              </button>
            </div>
          )}

          {/* Visualize button with dropdown */}
          <button
            className="flex items-center px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-blue-600 mb-2 w-full"
            onClick={toggleVisualizationDropdown}
          >
            <ChartBar className="w-4 h-4 mr-2" />
            Visualize
          </button>
          {visualizationDropdownVisible && (
            <div className="bg-gray-200 rounded-lg shadow-lg p-2 mt-2 mb-2">
              {/* <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createChart')}>
                Create Chart
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createGraph')}>
                Create Graph
              </button> */}
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createBar')}>
                Bar Plot
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createLine')}>
                Line Plot
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createViolin')}>
                Violin Plot
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createScatter')}>
                Scatter Plot
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createSilhouette')}>
                Silhouette Plot
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleVisualizationChange('createHeatMap')}>
                Heat Map
              </button>
            </div>
          )}

          {/* Machine Learning button with dropdown */}
          <button
            className="flex items-center px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-blue-600 mb-2 w-full"
            onClick={toggleMlDropdown}
          >
            <Brain className="w-4 h-4 mr-2" />
            Machine Learning
          </button>
          {mlDropdownVisible && (
            <div className="bg-gray-200 rounded-lg shadow-lg p-2 mt-2 mb-2">
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleMlChange('linearRegression')}>
                Linear Regression
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleMlChange('logisticRegression')}>
                Logistic Regression
              </button>
              <button className="block w-full px-4 py-2 text-left" onClick={() => handleMlChange('randomForest')}>
                Random Forest
              </button>
            </div>
          )}
        </div>

        <div className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onConnect={onConnect}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeDoubleClick={onNodeDoubleClick}
          >
            <Controls />
            <Background />
            {isDialogOpen && selectedNodeData.type === 'data' && (
              <div
                className="rounded-lg fixed top-1/2 left-1/2 bg-white rounded shadow-lg border p-4 w-[400px] h-[400px] flex flex-col justify-center items-center transform -translate-x-1/2 -translate-y-1/2 z-50"
              >
                {/* <h2 className="text-xl font-bold mb-4">Node Details</h2>
              <p className="mb-4">{dialogContent}</p> */}

                {/* Center the file input and button */}
                <div className="flex flex-col items-center justify-center mb-4">
                  <input
                    type="file"
                    onChange={handleFileUpload}
                    className="mb-2"
                    accept={
                      dialogContent === "CSV File" ? ".csv" :
                        dialogContent === "SQL Table" ? ".sql" :
                          // dialogContent === "Text Node" ? ".txt" : 
                          "*/*" // Default: Accept all files
                    }
                  />
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mt-2"
                    onClick={uploadFile}
                  >
                    Upload {dialogContent}
                  </button>
                </div>

                {/* Conditional rendering to display received data */}
                {data && (
                  <div className="mt-4 p-2 border rounded bg-gray-100 w-full text-center">
                    Data received: {JSON.stringify(data)}
                  </div>
                )}

                <button
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 mt-2"
                  onClick={closeDialog}
                >
                  Close
                </button>
              </div>
            )},

            {isDialogOpen && selectedNodeData.data.label === 'Impute' && (
              <div
                className="rounded-lg fixed top-1/2 left-1/2 bg-white rounded shadow-lg border p-4 flex flex-col justify-center items-center transform -translate-x-1/2 -translate-y-1/2 z-50"
              >
                <div className="flex flex-col items-center justify-center mb-4">
                  {/* Removed the "Enter value" input field */}
                  
                  <label className="font-medium mb-2">
                    Select columns to impute:
                  </label>
                  <div className="flex flex-col items-center justify-center mb-4 max-h-60 overflow-y-auto border border-gray-200 rounded">
                    <AnalyzeHeadersGrid 
                      onHeaderSelect={(header) => {
                        console.log(`Header ${header} selected`);
                      }} 
                    />
                  </div>
                  
                  <label htmlFor="strategyDrop" className="font-medium mb-2">
                    Select imputation strategy:
                  </label>
                  <select
                    id="strategyDrop"
                    className="mb-4 border border-gray-300 rounded px-2 py-1"
                  >
                    <option value="mean">Mean</option>
                    <option value="median">Median</option>
                    <option value="most_frequent">Most Frequent</option>
                    <option value="constant">Constant</option>
                  </select>
                  
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mt-2"
                    onClick={dataImputer}
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
            )},
            {isDialogOpen && selectedNodeData.data.label === 'Standardize Data' && (
              <div
                className="rounded-lg fixed top-1/2 left-1/2 bg-white rounded shadow-lg border p-4 flex flex-col justify-center items-center transform -translate-x-1/2 -translate-y-1/2 z-50"
              >
                <div className="flex flex-col items-center justify-center mb-4">
                  <label className="font-medium mb-2">
                    Select columns to standardize:
                  </label>
                  <div className="flex flex-col items-center justify-center mb-4 max-h-60 overflow-y-auto border border-gray-200 rounded">
                    <AnalyzeHeadersGrid 
                      onHeaderSelect={(header) => {
                        console.log(`Header ${header} selected for standardization`);
                        // This callback will be triggered when a checkbox is clicked
                      }} 
                    />
                  </div>
                  
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mt-2"
                    onClick={standardizeData}
                  >
                    Standardize
                  </button>
                </div>

                <button
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 mt-2"
                  onClick={closeDialog}
                >
                  Close
                </button>
              </div>
            )},
            {isDialogOpen && selectedNodeData && selectedNodeData.type === 'visualize' && (
              <div
                className="rounded-lg fixed top-1/2 left-1/2 bg-white rounded shadow-lg border p-4 flex flex-col justify-center items-center transform -translate-x-1/2 -translate-y-1/2 z-50"
              >
                <div className="flex flex-col items-center justify-center mb-4">
                  <h2 className="text-xl font-bold mb-4">{dialogContent}</h2>
                  
                  <label className="font-medium mb-2">
                    Select columns for visualization:
                  </label>
                  <div className="flex flex-col items-center justify-center mb-4 max-h-60 overflow-y-auto border border-gray-200 rounded">
                    <AnalyzeHeadersGrid 
                      onHeaderSelect={(header) => {
                        console.log(`Header ${header} selected for visualization`);
                      }} 
                    />
                  </div>
                  
                  {/* Add specific instructions based on plot type */}
                  {dialogContent === 'Scatter Plot' && (
                    <p className="text-sm text-gray-600 mb-2">
                      Please select at least 2 columns for a scatter plot. The first 2 selected columns will be used for X and Y axes.
                    </p>
                  )}
                  {dialogContent === 'Heat Map' && (
                    <p className="text-sm text-gray-600 mb-2">
                      Select multiple columns to include in the correlation heatmap.
                    </p>
                  )}
                  
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mt-2"
                    onClick={createVisualizationWithColumns}
                  >
                    Create {dialogContent}
                  </button>
                </div>

                <button
                  className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 mt-2"
                  onClick={closeDialog}
                >
                  Close
                </button>
              </div>
            )},
            {isDialogOpen && selectedNodeData && selectedNodeData.type === 'ml' && (
              <div
                className="rounded-lg fixed top-1/2 left-1/2 bg-white rounded shadow-lg border p-4 w-[500px] max-h-[80vh] overflow-y-auto flex flex-col transform -translate-x-1/2 -translate-y-1/2 z-50"
              >
                <h2 className="text-xl font-bold mb-4">{dialogContent} Configuration</h2>
                
                <div className="flex flex-col items-center justify-center mb-4">
                  <label className="font-medium mb-2">
                    Select feature columns:
                  </label>
                  <div className="flex flex-col items-center justify-center mb-4 max-h-60 overflow-y-auto border border-gray-200 rounded">
                    <AnalyzeHeadersGrid 
                      onHeaderSelect={(header) => {
                        console.log(`Header ${header} selected for ML features`);
                      }} 
                    />
                  </div>
                </div>
                
                <div className="mb-4">
                  <label htmlFor="targetColumn" className="font-medium block mb-2">
                    Select Target Column:
                  </label>
                  <select
                    id="targetColumn"
                    className="w-full border border-gray-300 rounded px-2 py-1"
                  >
                    <option value="">-- Select Target --</option>
                    {selectedNodeData.headers && selectedNodeData.headers.length > 0 ? (
                      selectedNodeData.headers.map((header, index) => (
                        <option key={index} value={header}>
                          {header}
                        </option>
                      ))
                    ) : (
                      <option value="" disabled>Loading columns...</option>
                    )}
                  </select>
                </div>
                
                <div className="mb-4">
                  <label htmlFor="testSize" className="font-medium block mb-2">
                    Test Size (0.1 - 0.5):
                  </label>
                  <input
                    type="number"
                    id="testSize"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    defaultValue="0.2"
                    className="w-full border border-gray-300 rounded px-2 py-1"
                  />
                </div>
                
                {dialogContent === 'Random Forest' && (
                  <>
                    <div className="mb-4">
                      <label htmlFor="taskType" className="font-medium block mb-2">
                        Task Type:
                      </label>
                      <select
                        id="taskType"
                        className="w-full border border-gray-300 rounded px-2 py-1"
                      >
                        <option value="classification">Classification</option>
                        <option value="regression">Regression</option>
                      </select>
                    </div>
                    
                    <div className="mb-4">
                      <label htmlFor="nEstimators" className="font-medium block mb-2">
                        Number of Estimators:
                      </label>
                      <input
                        type="number"
                        id="nEstimators"
                        min="10"
                        max="500"
                        step="10"
                        defaultValue="100"
                        className="w-full border border-gray-300 rounded px-2 py-1"
                      />
                    </div>
                  </>
                )}
                
                <div className="flex justify-between mt-4">
                  <button
                    className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                    onClick={closeDialog}
                  >
                    Cancel
                  </button>
                  <button
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    onClick={runMlAlgorithm}
                  >
                    Run {dialogContent}
                  </button>
                </div>
              </div>
            )},
            <ImputeComponent />
          </ReactFlow>
        </div>

        {/* Positioned buttons at the bottom right */}
        <div
          className="absolute bottom-4 right-4 flex flex-col space-y-2"
        >
          <button
            className="flex items-center px-4 py-2 bg-zinc-500 text-white rounded-lg hover:bg-red-600"
            onClick={deleteSelectedNode}
          >
            <Trash className="w-4 h-4 mr-2" />
            Delete Node
          </button>

          <button
            className="flex items-center px-4 py-2 bg-zinc-500 text-white rounded-lg hover:bg-red-600"
            onClick={deleteSelectedEdge}
          >
            <Trash className="w-4 h-4 mr-2" />
            Delete Edge
          </button>
        </div>

        {showPlotModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-4xl max-h-[90vh] overflow-auto">
               <div className="flex justify-between items-center mb-4">
                 <h2 className="text-xl font-bold">{currentPlot}</h2>
                 <button 
                   onClick={() => setShowPlotModal(false)}
                   className="text-gray-500 hover:text-gray-700"
                 >
                   ✕
                 </button>
                </div>
                <div className="flex justify-center">
                  <img 
                    src={plotUrl} 
                    alt={currentPlot} 
                    className="max-w-full max-h-[70vh] object-contain"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='6' x2='6' y2='18'%3E%3C/line%3E%3Cline x1='6' y1='6' x2='18' y2='18'%3E%3C/line%3E%3C/svg%3E";
                      alert("Plot not available. Please make sure you've uploaded data first.");
                    }}
                  />
                </div>
            </div>
        </div>
     )}

        {/* Add ML Results Modal */}
        {showMlResultsModal && mlResults && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-3xl max-h-[80vh] overflow-y-auto">
              <h2 className="text-xl font-bold mb-4">{dialogContent} Results</h2>
              
              {/* Display accuracy if available */}
              {mlResults.metrics && mlResults.metrics.accuracy && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">Accuracy</h3>
                  <div className="bg-gray-100 p-3 rounded">
                    <p className="text-xl">{mlResults.metrics.accuracy.toFixed(4)}</p>
                  </div>
                </div>
              )}
              
              {/* Display MSE and R2 for regression */}
              {mlResults.metrics && mlResults.metrics.mse && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">Regression Metrics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-100 p-3 rounded">
                      <p className="font-medium">MSE</p>
                      <p className="text-xl">{mlResults.metrics.mse.toFixed(4)}</p>
                    </div>
                    <div className="bg-gray-100 p-3 rounded">
                      <p className="font-medium">R²</p>
                      <p className="text-xl">{mlResults.metrics.r2.toFixed(4)}</p>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Display feature importance if available */}
              {mlResults.feature_importance && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">Feature Importance</h3>
                  <div className="bg-gray-100 p-3 rounded">
                    <ul className="divide-y">
                      {Object.entries(mlResults.feature_importance)
                        .sort((a, b) => b[1] - a[1])
                        .map(([feature, importance]) => (
                          <li key={feature} className="py-2 flex justify-between">
                            <span>{feature}</span>
                            <span className="font-medium">{importance.toFixed(4)}</span>
                          </li>
                        ))}
                    </ul>
                  </div>
                </div>
              )}
              
              {/* Display confusion matrix if available */}
              {mlResults.confusion_matrix && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">Confusion Matrix</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full bg-gray-100 rounded">
                      <tbody>
                        {Array.isArray(mlResults.confusion_matrix) && mlResults.confusion_matrix.map((row, i) => (
                          <tr key={i}>
                            {Array.isArray(row) && row.map((cell, j) => (
                              <td key={j} className="border p-2 text-center">{cell}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              
              {/* Display classification report if available */}
              {mlResults.classification_report && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold mb-2">Classification Report</h3>
                  <pre className="bg-gray-100 p-3 rounded overflow-x-auto whitespace-pre-wrap">
                    {typeof mlResults.classification_report === 'string' 
                      ? mlResults.classification_report 
                      : JSON.stringify(mlResults.classification_report, null, 2)}
                  </pre>
                </div>
              )}
              
              <div className="flex justify-end mt-4">
                <button
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                  onClick={() => setShowMlResultsModal(false)}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </ReactFlowProvider>
  );
};

export default WorkflowCanvas;
// Add this function to your component
const debugFetchHeaders = async () => {
  try {
    console.log("Fetching headers...");
    const response = await fetch('http://localhost:5001/headers');
    console.log("Response status:", response.status);
    if (response.ok) {
      const data = await response.json();
      console.log("Headers data:", data);
      // Replace this line that uses setHeaders
      // setHeaders(data.headers || []);
      
      // Instead, log the headers to console only
      console.log("Headers received:", data.headers || []);
      
      // Or if you need to store them somewhere, use localStorage temporarily
      localStorage.setItem('dataHeaders', JSON.stringify(data.headers || []));
    } else {
      console.error("Failed to fetch headers:", response.statusText);
    }
  } catch (error) {
    console.error('Error fetching headers:', error);
  }
};

// Add this button somewhere in your component for testing
// <button onClick={debugFetchHeaders}>Debug: Fetch Headers</button>
