# Financial Insight Dashboard
This project provides a full-stack solution for generating financial insights by leveraging advanced NLP models and financial data APIs. The backend is built with Flask to handle data processing and API management, while the frontend is developed using React to provide a responsive user interface.

## Features
Financial Data Retrieval: Integrates with APIs like Alpha Vantage to fetch real-time and historical stock data, earnings call transcripts, and SEC filings.
Insight Generation: Utilizes language models (e.g., Bert, Mistral) and a custom Retrieval-Augmented Generation (RAG) system for generating insights.
Interactive Dashboard: Allows users to query financial data and view generated insights through a user-friendly web interface.
Responsive Design: Ensures that the web application is accessible on a wide range of devices, from desktops to mobile phones.

## Setup Instructions
### Backend
The backend is developed using Flask. To get it up and running:

1. **Install Python Dependencies**:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

2. **Environment Variables**:
    - Set necessary environment variables or use a `.env` file to load configurations (API keys, model paths, etc.).

3. **Running the Flask Server**:
    ```bash
    export FLASK_APP=app.py
    flask run
    ```

### Frontend

The frontend is built with React. To start the frontend server:

1. **Install Node.js Dependencies**:
    ```bash
    cd frontend
    npm install
    ```

2. **Starting the Development Server**:
    ```bash
    npm start
    ```

## Usage
After setting up the backend and frontend:

Navigate to http://localhost:3000 on your web browser to access the dashboard.
Enter queries related to financial data, such as specific company insights or market trends.
View generated responses and data visualizations directly on the dashboard.
Contributing
Contributions to the project are welcome! Here are a few ways you can help:

Feature Enhancements: Propose new features or improvements to the current system.
Bug Reports: Report issues and bugs in the issue tracker.
Documentation: Improve or suggest enhancements to documentation.

## Project Report
[View the Project Report](./Report.pdf)
