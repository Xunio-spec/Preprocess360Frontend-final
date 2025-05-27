// import React from 'react';
// import WorkflowCanvas from './WorkflowCanvas';

// const App = () => {
//   return (
//     <div className="App">
//       <WorkflowCanvas />
//     </div>
//   );
// };

// export default App;


import React from 'react';
import { ReactFlowProvider } from 'reactflow';  // Ensure this is imported
import WorkflowCanvas from './WorkflowCanvas';  // Import your WorkflowCanvas

function App() {
  return (
    <ReactFlowProvider>  {/* Wrap your entire app with ReactFlowProvider */}
      <WorkflowCanvas />
    </ReactFlowProvider>
  );
}

export default App;
