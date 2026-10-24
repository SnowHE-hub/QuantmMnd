import React, { useState } from 'react';
import { StorageTreemap } from './StorageTreemap';
import { SchemaViewer } from './SchemaViewer';
import { DatasetGrid } from './DatasetGrid';
import { DataLineageGraph } from './DataLineageGraph';
import { CatalogData, CatalogEntry } from '../../api/types';

interface DataObservatoryViewProps {
  catalog: CatalogData | null;
  selectedEntryId?: string;
  onSelectEntryId?: (id: string) => void;
}

export const DataObservatoryView: React.FC<DataObservatoryViewProps> = ({
  catalog,
  selectedEntryId,
  onSelectEntryId,
}) => {
  const entries = catalog?.entries || [];

  const [activeEntry, setActiveEntry] = useState<CatalogEntry | null>(() => {
    if (selectedEntryId) {
      return entries.find((e) => e.id === selectedEntryId) || entries[0] || null;
    }
    return entries[0] || null;
  });

  const handleSelect = (entry: CatalogEntry) => {
    setActiveEntry(entry);
    if (onSelectEntryId) onSelectEntryId(entry.id);
  };

  return (
    <div className="space-y-4">
      {/* 1. Storage Treemap */}
      <StorageTreemap
        entries={entries}
        onSelectEntry={handleSelect}
        selectedEntryId={activeEntry?.id}
      />

      {/* 2. Interactive Schema & Sampling Inspector */}
      <SchemaViewer entry={activeEntry} />

      {/* 3. Data Processing Lineage Graph */}
      <DataLineageGraph />

      {/* 4. Full 44 Datasets Catalog Grid */}
      <DatasetGrid
        entries={entries}
        selectedEntryId={activeEntry?.id}
        onSelectEntry={handleSelect}
      />
    </div>
  );
};
