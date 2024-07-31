// NewPage.js

import React, { useState } from 'react';
import './NewPage.css'; // Import your CSS file
import { Link } from 'react-router-dom';
import QuerySender from './QuerySender'; // Import the QuerySender component
import QuerySender2 from './QuerySender2'; // Import the QuerySender component

function NewPage() {
  const [selectedStock, setSelectedStock] = useState(''); // State to store the selected stock
  const [message, setMessage] = useState(''); // State to store the message

  const handleStockChange = (event) => {
    setSelectedStock(event.target.value); // Update the selected stock state when dropdown value changes
  };

  const handleMessageChange = (event) => {
    setMessage(event.target.value); // Update the message state as the input changes
  };

  const sendMessage = () => {
    // Print the selected stock and message in the console
    console.log("Selected Stock:", selectedStock);
    console.log("Message:", message);

    // You can add further logic here to send the selected stock and message to a server or other components
    
    // Optionally, reset the selected stock and message states after sending
    //setSelectedStock('');
    //setMessage('');
  };

  // List of Nifty 50 stocks
  const nifty50Stocks = [
    "Adani Ports and Special Economic Zone",
    "Asian Paints",
    "Axis Bank",
    "Bajaj Auto",
    "Bajaj Finance",
    "Bajaj Finserv",
    "Bharti Airtel",
    "Britannia Industries",
    "Cipla",
    "Coal India",
    "Divi's Laboratories",
    "Dr Reddy's Laboratories",
    "Eicher Motors",
    "Grasim Industries",
    "HCL Technologies",
    "HDFC",
    "HDFC Bank",
    "HDFC Life Insurance",
    "Hero MotoCorp",
    "Hindalco Industries",
    "Hindustan Unilever",
    "ICICI Bank",
    "Indian Oil Corporation",
    "IndusInd Bank",
    "Infosys",
    "IOC",
    "ITC",
    "JSW Steel",
    "Kotak Mahindra Bank",
    "Larsen & Toubro",
    "Mahindra & Mahindra",
    "Marico",
    "Maruti Suzuki India",
    "Nestlé India",
    "NTPC",
    "Oil and Natural Gas Corporation",
    "Power Grid Corporation of India",
    "Reliance Industries",
    "Shree Cement",
    "State Bank of India",
    "SUN PHARMA",
    "Tata Consumer Products",
    "Tata Motors",
    "Tata Steel",
    "Tech Mahindra",
    "Titan Company",
    "UltraTech Cement",
    "UPL",
    "Wipro"
  ];

  return (
    <div className="NewPage">
      <div className="left-section">
        {/* Dropdown menu for Nifty-50 stocks */}
        <div className='hhh'>
        <select value={selectedStock} onChange={handleStockChange}>
          <option value="">Choose from Nifty-50</option>
          {nifty50Stocks.map(stock => (
            <option key={stock} value={stock}>{stock}</option>
          ))}
        </select>
        </div>
        <div className="aa">
          <QuerySender2 message={selectedStock} />
        </div>
      </div>
      <div className="chat-section">
        {/* Chat history section */}
        <div className="chat-history">
          {/* Chat messages will be rendered here */}
        </div>
        {/* Text input box */}
        <div className="text-input">
          <input 
            type="text" 
            placeholder="Type your message..." 
            value={message} // Bind input value to message state
            onChange={handleMessageChange} // Handle input change
          />
          <button onClick={() => sendMessage()}>Send</button> {/* Call sendMessage when the button is clicked */}
        </div>
      </div>
      {/* Pass the selected stock as a prop to QuerySender */}
      <QuerySender message={message} />
    </div>

  );
}

export default NewPage;
