'use client';

import { Bar, Doughnut, Line, Pie } from 'react-chartjs-2';
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
} from 'chart.js';
import type { Visual, VisualDataset } from '../hooks/useChat';
import { useMemo, useState } from 'react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Tooltip, Legend, Title);

const palette = [
  '#4F46E5',
  '#22D3EE',
  '#F59E0B',
  '#10B981',
  '#EF4444',
  '#8B5CF6',
  '#0EA5E9',
];

function normalizeDatasets(raw?: VisualDataset[] | any): VisualDataset[] {
  if (!raw) return [];
  const arr = Array.isArray(raw) ? raw : [raw];
  return arr.map((ds, idx) => {
    if (typeof ds !== 'object' || ds === null) return { label: `Series ${idx + 1}`, data: [] };
    return {
      label: ds.label || `Series ${idx + 1}`,
      data: Array.isArray(ds.data) ? ds.data.map((n: any) => Number(n)) : [],
      backgroundColor: ds.backgroundColor || palette[idx % palette.length],
      borderColor: ds.borderColor || palette[idx % palette.length],
      fill: ds.fill,
    };
  });
}

function ChartBlock({ visual }: { visual: Visual }) {
  const datasets = useMemo(() => normalizeDatasets(visual.datasets), [visual.datasets]);
  const labels = Array.isArray(visual.labels) ? visual.labels : [];
  const title = visual.title || (visual.type ? String(visual.type).toUpperCase() : undefined);

  const data = { labels, datasets };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      // Keep visuals low-profile in chat by hiding large legends by default
      legend: { display: false },
      title: { display: false },
    },
  };

  switch (visual.type) {
    case 'bar':
      return <Bar data={data} options={options} />;
    case 'pie':
      return <Pie data={data} options={options} />;
    case 'doughnut':
      return <Doughnut data={data} options={options} />;
    case 'line':
    default:
      return <Line data={data} options={options} />;
  }
}

function TableBlock({ visual }: { visual: Visual }) {
  const columns = Array.isArray(visual.columns) && visual.columns.length ? visual.columns : undefined;
  const rows = Array.isArray(visual.rows) ? visual.rows : [];
  const inferredColumns = !columns && rows.length && typeof rows[0] === 'object' && rows[0] !== null && !Array.isArray(rows[0])
    ? Object.keys(rows[0])
    : [];
  const cols = columns || inferredColumns;

  const [showAll, setShowAll] = useState(false);
  const rowsToShow = showAll ? rows : rows.slice(0, 5);

  // Check if rows are arrays (positional) or objects (keyed)
  const isArrayRows = rows.length > 0 && Array.isArray(rows[0]);

  return (
    <div className="visual-table">
      {visual.title && <div className="text-sm font-semibold mb-2 text-white/80">{visual.title}</div>}
      {visual.description && <div className="text-xs mb-2 text-white/60">{visual.description}</div>}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm text-white/80 border-collapse">
          <thead>
            <tr className="border-b border-white/10">
              {(cols || []).map((col, colIdx) => (
                <th key={colIdx} className="text-left py-2 pr-4 font-semibold">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rowsToShow.map((row, idx) => (
              <tr key={idx} className="border-b border-white/5">
                {isArrayRows 
                  ? (row as any[]).map((cell: any, cellIdx: number) => (
                      <td key={cellIdx} className="py-2 pr-4 text-white/70">{String(cell ?? '')}</td>
                    ))
                  : (cols || []).map((col) => (
                      <td key={col} className="py-2 pr-4 text-white/70">{typeof row === 'object' && row !== null ? String(row[col] ?? '') : ''}</td>
                    ))
                }
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > 5 && (
        <div className="mt-2 text-xs text-white/60">
          Showing top 5 rows. <button className="link-button" onClick={() => setShowAll(!showAll)}>{showAll ? 'Show less' : 'View full table'}</button>
        </div>
      )}
    </div>
  );
}

export default function VisualRenderer({ visual }: { visual: Visual }) {
  if (!visual) return null;

  if (visual.type === 'table') {
    return <TableBlock visual={visual} />;
  }

  return (
    <div className="visual-chart">
      {/* caption shown by MessageBubble; keep description minimal */}
      {visual.description && <div className="text-xs mb-1 text-white/60">{visual.description}</div>}
      <div style={{ height: 150 }}>
        <ChartBlock visual={visual} />
      </div>
    </div>
  );
}
