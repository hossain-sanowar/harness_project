// src/components/MetricsTable.js
import React from 'react';
import { Table } from 'antd';

const MetricsTable = ({ data }) => {
  const columns = [
    {
      title: 'Metric',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: 'Value',
      dataIndex: 'value',
      key: 'value',
      render: (text, record) => {
        if (record.unit) return `${text} ${record.unit}`;
        return text;
      }
    },
  ];

  const tableData = [
    { key: '1', metric: 'Number of components per cluster', value: data.componentsPerCluster },
    { key: '2', metric: 'Components to centroid distance', value: data.compToCentroidDist, unit: 'm' },
    { key: '3', metric: 'Components to centroid weight', value: data.compToCentroidWeight, unit: 'g' },
    { key: '4', metric: 'Centroid to centroid distance', value: data.centroidToCentroidDist, unit: 'm' },
    { key: '5', metric: 'Centroid to centroid weight', value: data.centroidToCentroidWeight, unit: 'g' },
    { key: '6', metric: 'Centroid to battery distance', value: data.centroidToBatteryDist, unit: 'm' },
    { key: '7', metric: 'Centroid to battery weight', value: data.centroidToBatteryWeight, unit: 'g' },
    { key: '8', metric: 'Centroid to HPC distance', value: data.centroidToHpcDist, unit: 'm' },
    { key: '9', metric: 'Centroid to HPC weight', value: data.centroidToHpcWeight, unit: 'g' },
    { key: '10', metric: 'Cluster distance', value: data.clusterDistance, unit: 'm' },
    { key: '11', metric: 'Cluster weight', value: data.clusterWeight, unit: 'g' },
    { key: '12', metric: 'Total harness distance', value: data.totalHarnessDist, unit: 'm' },
    { key: '13', metric: 'Total harness weight', value: data.totalHarnessWeight, unit: 'kg' },
  ];

  return (
    <Table 
      columns={columns} 
      dataSource={tableData} 
      pagination={false}
      size="small"
      bordered
      className="metrics-table"
    />
  );
};

export default MetricsTable;