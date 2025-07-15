// src/components/LiveTraining.js
import React, { useState } from 'react';
import { Card, Form, InputNumber, Select, Checkbox, Button, Row, Col, Tabs } from 'antd';
import HarnessVisualization from './HarnessVisualization';
import MetricsTable from './MetricsTable';
import './LiveTraining.css';

const { TabPane } = Tabs;
const { Option } = Select;

const LiveTraining = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [activeVizTab, setActiveVizTab] = useState('1');

  const onFinish = async (values) => {
    setLoading(true);
    // API call would go here
    setTimeout(() => {
      setResults({
        images: {
          withoutRouting: '/without-routing.png',
          restrictedZone: '/restricted-zone.png',
          withRouting: '/with-routing.png'
        },
        metrics: {
          componentsPerCluster: 8,
          compToCentroidDist: 1.2,
          compToCentroidWeight: 320,
          centroidToCentroidDist: 4.5,
          centroidToCentroidWeight: 780,
          centroidToBatteryDist: 3.2,
          centroidToBatteryWeight: 650,
          centroidToHpcDist: 2.8,
          centroidToHpcWeight: 580,
          clusterDistance: 12.5,
          clusterWeight: 2350,
          totalHarnessDist: 45.8,
          totalHarnessWeight: 8.2
        }
      });
      setLoading(false);
    }, 2000);
  };

  return (
    <Card title="Live AI Model Training" bordered={false}>
      <Row gutter={[24, 24]}>
        <Col span={8}>
          <Form
            form={form}
            layout="vertical"
            onFinish={onFinish}
            initialValues={{
              application: 'mid',
              clusters: 5,
              wireType: 'can-fd',
              dynamicDriving: true
            }}
          >
            <Form.Item 
              name="application" 
              label="Application Level"
              rules={[{ required: true }]}
            >
              <Select>
                <Option value="low">Low</Option>
                <Option value="mid">Mid</Option>
                <Option value="high">High</Option>
              </Select>
            </Form.Item>
            
            <Form.Item 
              name="clusters" 
              label="Number of Clusters"
              rules={[{ required: true }]}
            >
              <InputNumber min={1} max={14} style={{ width: '100%' }} />
            </Form.Item>
            
            <Form.Item 
              name="wireType" 
              label="Wire Type"
              rules={[{ required: true }]}
            >
              <Select>
                <Option value="can-fd">CAN-FD (FLRY 2×0.35)</Option>
                <Option value="ethernet">Ethernet (FLKS9Y2x0.13)</Option>
                <Option value="power">Power Cable (FLRY 1.5mm²)</Option>
              </Select>
            </Form.Item>
            
            <Form.Item 
              name="batteryPosition" 
              label="Battery Position"
              rules={[{ required: true }]}
            >
              <Select>
                <Option value="front">Front</Option>
                <Option value="middle">Middle</Option>
                <Option value="rear">Rear</Option>
              </Select>
            </Form.Item>
            
            <Form.Item name="dynamicDriving" valuePropName="checked">
              <Checkbox>Include Dynamic Driving Scenarios</Checkbox>
            </Form.Item>
            
            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading}
                block
              >
                Train Model
              </Button>
            </Form.Item>
          </Form>
        </Col>
        
        <Col span={16}>
          {results ? (
            <>
              <Tabs 
                activeKey={activeVizTab} 
                onChange={setActiveVizTab}
                className="visualization-tabs"
              >
                <TabPane tab="Without Routing" key="1">
                  <HarnessVisualization image={results.images.withoutRouting} />
                </TabPane>
                <TabPane tab="Avoid Restricted Zones" key="2">
                  <HarnessVisualization image={results.images.restrictedZone} />
                </TabPane>
                <TabPane tab="With Routing" key="3">
                  <HarnessVisualization image={results.images.withRouting} />
                </TabPane>
              </Tabs>
              
              <MetricsTable data={results.metrics} />
            </>
          ) : (
            <div className="results-placeholder">
              <div className="placeholder-image" />
              <p>Configure parameters and train model to see results</p>
            </div>
          )}
        </Col>
      </Row>
    </Card>
  );
};

export default LiveTraining;