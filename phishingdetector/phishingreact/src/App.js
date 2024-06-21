import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

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
                      onChange={(e) => setEmailSubject(e.target.value)}
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
                  <button type="submit" disabled={emailBody.length < 150}>Check Email</button>
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
                      }}
                      placeholder="Enter URL here..."
                    />
                    {urlError && <p className="error-message">{urlError}</p>}
                  </div>
                  <button type="submit">Check URL</button>
                </form>
              </div>
            )}
          </div>
          <div className="results-section">
            <h2>Results</h2>
            {activeTab === 'email' && renderResult(emailPrediction, emailProbability, 'email')}
            {activeTab === 'url' && renderResult(urlPrediction, urlProbability, 'url')}
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