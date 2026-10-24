import React, { useState } from 'react';
import { FormalPortfolio } from './FormalPortfolio';
import { OrdersNextOpen } from './OrdersNextOpen';
import { BrainRadar } from './BrainRadar';
import { MarketBarometer } from './MarketBarometer';
import { CandidatesFunnel } from './CandidatesFunnel';
import { BypassEvents } from './BypassEvents';
import { TruthTestPanel } from './TruthTestPanel';
import {
  FormalReport,
  BrainResponse,
  MarketState,
  CandidateItem,
  BypassEvent,
  RiskCardItem,
} from '../../api/types';

interface CockpitViewProps {
  formal: FormalReport | null;
  brain: BrainResponse | null;
  market: MarketState | null;
  candidates: CandidateItem[];
  bypassEvents: BypassEvent[];
  riskCards?: RiskCardItem[];
  horizon: '20d' | '63d';
  onChangeHorizon: (h: '20d' | '63d') => void;
  onSelectTicker: (ticker: string) => void;
  onAddToCompare?: (ticker: string) => void;
  selectedDate: string;
  latestDate: string;
}

export const CockpitView: React.FC<CockpitViewProps> = ({
  formal,
  brain,
  market,
  candidates,
  bypassEvents,
  riskCards = [],
  horizon,
  onChangeHorizon,
  onSelectTicker,
  onAddToCompare,
  selectedDate,
  latestDate,
}) => {
  const [selectedTierKey, setSelectedTierKey] = useState<string>('aum_1e5');

  // Extract orders dynamically based on selected tier
  const activeTier = formal?.tiers?.[selectedTierKey];
  const orders = activeTier?.orders_next_open || [];

  return (
    <div className="space-y-4">
      {/* 1. Formal Recommendations & 3 AUM Tiers */}
      <FormalPortfolio
        formal={formal}
        onSelectTicker={onSelectTicker}
        selectedTierKey={selectedTierKey}
        onSelectTierKey={setSelectedTierKey}
      />

      {/* 2. Tomorrow's Open Orders (with odd-lot warning, Red Team Guard, and Sandbox) */}
      <OrdersNextOpen
        orders={orders}
        tier={activeTier}
        brainFlags={brain?.memo?.flags || []}
        riskCards={riskCards}
        selectedDate={selectedDate}
        latestDate={latestDate}
        onSelectTicker={onSelectTicker}
        onAddToCompare={onAddToCompare}
      />

      {/* 2b. 事后验真：模型 IC 与红队规则（后端统一计算） */}
      <TruthTestPanel selectedDate={selectedDate} />

      {/* 3. Market State Barometer */}
      <MarketBarometer market={market} />

      {/* 4. Agent Brain A0 Radar */}
      <BrainRadar brain={brain} onSelectTicker={onSelectTicker} />

      {/* 5. F4 Candidate Funnel */}
      <CandidatesFunnel
        candidates={candidates}
        horizon={horizon}
        onChangeHorizon={onChangeHorizon}
        onSelectTicker={onSelectTicker}
        onAddToCompare={onAddToCompare}
      />

      {/* 6. Bypass Events Stream */}
      <BypassEvents events={bypassEvents} onSelectTicker={onSelectTicker} />
    </div>
  );
};
