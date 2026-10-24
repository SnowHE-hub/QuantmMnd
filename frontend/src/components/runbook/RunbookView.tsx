import React from 'react';
import { PlaybookDeck } from './PlaybookDeck';
import { CommandGenerator } from './CommandGenerator';

interface RunbookViewProps {
  selectedDate: string;
  onNavigateTab: (tab: any) => void;
  onSelectTicker: (ticker: string) => void;
}

export const RunbookView: React.FC<RunbookViewProps> = ({
  selectedDate,
  onNavigateTab,
  onSelectTicker,
}) => {
  return (
    <div className="space-y-5">
      {/* 1. Playbook Deck */}
      <PlaybookDeck onNavigateTab={onNavigateTab} onSelectTicker={onSelectTicker} />

      {/* 2. Visual Parameter & CLI Generator */}
      <CommandGenerator selectedDate={selectedDate} />
    </div>
  );
};
