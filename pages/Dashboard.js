// src/pages/Dashboard.js
import React, { useState } from 'react';
import { Layout, Menu, theme, ConfigProvider } from 'antd';
import { 
  ExperimentOutlined, 
  DeploymentUnitOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import HeaderBar from '../components/HeaderBar';
import LiveTraining from '../components/LiveTraining';
import PreTrainedModel from '../components/PreTrainedModel';
import NetworkTopology from '../components/NetworkTopology';
import FormalVerification from '../components/FormalVerification';
import UserProfile from '../components/UserProfile';
import ImageCarousel from '../components/ImageCarousel';
import InfoPanel from '../components/InfoPanel';
import '../styles/Dashboard.css';

const { Content, Sider } = Layout;

const Dashboard = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('live-training');
  const {
    token: { colorBgContainer },
  } = theme.useToken();

  const renderContent = () => {
    switch(activeTab) {
      case 'live-training': return <LiveTraining />;
      case 'pre-trained': return <PreTrainedModel />;
      case 'topology': return <NetworkTopology />;
      case 'verification': return <FormalVerification />;
      case 'profile': return <UserProfile />;
      default: return <LiveTraining />;
    }
  };

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          borderRadius: 4,
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Sider 
          collapsible 
          collapsed={collapsed} 
          onCollapse={(value) => setCollapsed(value)}
          theme="light"
        >
          <div className="logo">
            <img src="/company-logo.png" alt="Company Logo" />
          </div>
          <Menu
            theme="light"
            mode="inline"
            selectedKeys={[activeTab]}
            onSelect={({key}) => setActiveTab(key)}
            items={[
              {
                key: 'live-training',
                icon: <ExperimentOutlined />,
                label: 'Live Training',
              },
              {
                key: 'pre-trained',
                icon: <DeploymentUnitOutlined />,
                label: 'Pre-trained Model',
              },
              {
                key: 'topology',
                icon: <DeploymentUnitOutlined />,
                label: 'Network Topology',
              },
              {
                key: 'verification',
                icon: <SafetyCertificateOutlined />,
                label: 'Formal Verification',
              },
              {
                key: 'profile',
                icon: <UserOutlined />,
                label: 'User Profile',
              },
              {
                key: 'logout',
                icon: <LogoutOutlined />,
                label: 'Logout',
                danger: true,
              }
            ]}
          />
        </Sider>
        
        <Layout>
          <HeaderBar />
          
          <Content
            style={{
              margin: '16px',
              padding: 24,
              minHeight: 280,
              background: colorBgContainer,
              position: 'relative'
            }}
          >
            {renderContent()}
            
            {/* Static Elements */}
            <ImageCarousel />
            <InfoPanel />
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
};

export default Dashboard;