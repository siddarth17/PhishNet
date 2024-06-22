import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import './App.css';

ChartJS.register(ArcElement, Tooltip, Legend);

function App() {
  const [activeTab, setActiveTab] = useState('email');
  const [emailSubject, setEmailSubject] = useState('');
  const [emailBody, setEmailBody] = useState('');
  const [emailBodyError, setEmailBodyError] = useState('');
  const [emailPrediction, setEmailPrediction] = useState(null);
  const [emailProbability, setEmailProbability] = useState(null);
  const [url, setUrl] = useState('');
  const [urlError, setUrlError] = useState('');
  const [urlPrediction, setUrlPrediction] = useState(null);
  const [urlProbability, setUrlProbability] = useState(null);
  const [emailStats, setEmailStats] = useState({ phishing: 0, not_phishing: 0 });
  const [urlStats, setUrlStats] = useState({ phishing: 0, not_phishing: 0 });
  const [isEmailSubmitted, setIsEmailSubmitted] = useState(false);
  const [isUrlSubmitted, setIsUrlSubmitted] = useState(false);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('http://localhost:5001/stats');
      setEmailStats(response.data.email);
      setUrlStats(response.data.url);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const handleEmailSubmit = async (e) => {
    e.preventDefault();
    if (emailBody.length < 150) {
      setEmailBodyError('Email body must be at least 150 characters long.');
      return;
    }
    setEmailBodyError('');
    const combinedText = `Subject: ${emailSubject}\n\nBody: ${emailBody}`;
    try {
      const response = await axios.post('http://localhost:5001/predict-email', { text: combinedText });
      setEmailPrediction(response.data.prediction);
      setEmailProbability(response.data.probability);
      setIsEmailSubmitted(true);
      fetchStats();
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5001/predict-url', { url });
      setUrlPrediction(response.data.prediction);
      setUrlProbability(response.data.probability);
      setIsUrlSubmitted(true);
      fetchStats();
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const renderResult = (prediction, probability, type) => {
    if (prediction === null) return null;

    let resultText, confidenceText;

    if (type === 'email') {
      resultText = prediction === 1 ? 'Phishing Email' : 'Not Phishing Email';
    } else {
      resultText = prediction === 1 ? 'Malicious URL' : 'Not Malicious URL';
    }

    if (prediction === 0) {
      confidenceText = `It is very likely that the ${type} is safe.`;
    } else {
      if (probability > 80) {
        confidenceText = `It is very likely the ${type} is malicious. Be careful!`;
      } else {
        confidenceText = `The ${type} is suspicious. There is a possibility that it is not safe!`;
      }
    }

    return (
      <div className="result">
        <h3>Prediction: {resultText}</h3>
        <p>{confidenceText}</p>
      </div>
    );
  };

  const renderPieChart = (stats, type) => {
    const data = {
      labels: ['Phishing', 'Not Phishing'],
      datasets: [
        {
          data: [stats.phishing, stats.not_phishing],
          backgroundColor: ['#FF6384', '#36A2EB'],
          hoverBackgroundColor: ['#FF6384', '#36A2EB']
        }
      ]
    };
  
    const options = {
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.label || '';
              const value = context.parsed || 0;
              const total = context.dataset.data.reduce((acc, data) => acc + data, 0);
              const percentage = ((value / total) * 100).toFixed(0);
              return `${label}: ${percentage}%`;
            }
          }
        },
        legend: {
          display: true,
          position: 'bottom'
        }
      }
    };
  
    const total = stats.phishing + stats.not_phishing;
    const phishingPercentage = Math.round((stats.phishing / total) * 100);
  
    return (
      <div className="stats">
        <h3>{type} Statistics</h3>
        <Pie data={data} options={options} />
        <p>{phishingPercentage}% Phishing {type}s detected</p>
      </div>
    );
  };

  return (
    <div className="App">
      <header>
        <h1>Phishnet</h1>
      </header>
      <main>
        <div className="tab-container">
          <button 
            className={activeTab === 'email' ? 'active' : ''}
            onClick={() => setActiveTab('email')}
          >
            Email
          </button>
          <button 
            className={activeTab === 'url' ? 'active' : ''}
            onClick={() => setActiveTab('url')}
          >
            URL
          </button>
        </div>
        <div className="content">
          <div className={`input-section ${activeTab === 'url' ? 'url-mode' : ''}`}>
            {activeTab === 'email' && (
              <div className="email-detection">
                <h2>Email Detection</h2>
                <form onSubmit={handleEmailSubmit}>
                  <div className="form-group">
                    <label htmlFor="emailSubject">Email Subject:</label>
                    <input
                      id="emailSubject"
                      type="text"
                      value={emailSubject}
                      onChange={(e) => {
                        setEmailSubject(e.target.value);
                        setIsEmailSubmitted(false);
                      }}
                      placeholder="Enter the email subject"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="emailBody">Email Body:</label>
                    <textarea
                      id="emailBody"
                      value={emailBody}
                      onChange={(e) => {
                        setEmailBody(e.target.value);
                        setIsEmailSubmitted(false);
                        if (e.target.value.length < 150) {
                          setEmailBodyError('Email body must be at least 150 characters long.');
                        } else {
                          setEmailBodyError('');
                        }
                      }}
                      placeholder="Enter the email body"
                      rows="6"
                    />
                    {emailBodyError && <p className="error-message">{emailBodyError}</p>}
                  </div>
                  <button type="submit" disabled={emailBody.length < 150 || isEmailSubmitted}>Check Email</button>
                </form>
              </div>
            )}
            {activeTab === 'url' && (
              <div className="url-detection">
                <h2>URL Detection</h2>
                <form onSubmit={handleUrlSubmit}>
                  <div className="form-group">
                    <label htmlFor="url">URL:</label>
                    <input
                      id="url"
                      type="text"
                      value={url}
                      onChange={(e) => {
                        setUrl(e.target.value);
                        setUrlError('');
                        setIsUrlSubmitted(false);
                      }}
                      placeholder="Enter URL here..."
                    />
                    {urlError && <p className="error-message">{urlError}</p>}
                  </div>
                  <button type="submit" disabled={isUrlSubmitted}>Check URL</button>
                </form>
              </div>
            )}
          </div>
          <div className="results-section">
            <h2>Results</h2>
            {activeTab === 'email' && renderResult(emailPrediction, emailProbability, 'email')}
            {activeTab === 'url' && renderResult(urlPrediction, urlProbability, 'url')}
            {activeTab === 'email' && renderPieChart(emailStats, 'Email')}
            {activeTab === 'url' && renderPieChart(urlStats, 'URL')}
          </div>
        </div>
      </main>
      <footer>
        <p>Disclaimer: This tool is for educational purposes only and does not guarantee complete accuracy.</p>
        <p className="citation">Citations: *Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619*</p>
      </footer>
    </div>
  );
}

export default App;