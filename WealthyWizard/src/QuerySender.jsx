// QuerySender.js
import React, { useState } from 'react';

function QuerySender({ message }) { // Accept message prop
  console.log("A1:", message);
  const [response, setResponse] = useState('');

  const sendQuery = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message }), // Use message prop as the query
      });
      const data = await res.json();
      setResponse(data.response);
      
      // Log the response in the console
      console.log('Response:', data.response);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      {/* Removed input for query */}
      <button onClick={sendQuery}>Send Query</button>
      
      <p>Response: {response}</p>
    </div>
  );
}

export default QuerySender;
