// src/components/InfoPanel.js
import React, { useState } from 'react';
import { Card } from 'antd';
import './InfoPanel.css';

const InfoPanel = () => {
  const [infoIndex, setInfoIndex] = useState(0);
  
  const infoItems = [
    { title: 'Design Engineer', content: 'John Smith, Senior Harness Designer' },
    { title: 'Contact', content: 'john.smith@company.com' },
    { title: 'Last Updated', content: 'July 15, 2025' },
    { title: 'Current Project', content: 'EV-2025 Harness System' }
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setInfoIndex(prev => (prev + 1) % infoItems.length);
    }, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="corner-info bottom-left">
      <Card 
        title={infoItems[infoIndex].title} 
        size="small"
        className="info-card"
      >
        <p>{infoItems[infoIndex].content}</p>
      </Card>
    </div>
  );
};

export default InfoPanel;