import React from 'react';
import { StatsContainer, StatCard, StatValue, StatLabel } from './StyledComponents';

export default function StatisticsPanel({ statistics }) {
  if (!statistics) return null;

  return (
    <StatsContainer>
      <StatCard>
        <StatValue>{statistics.total_pieces}</StatValue>
        <StatLabel>Total Pieces</StatLabel>
      </StatCard>
      <StatCard>
        <StatValue>{statistics.annotation_completion_rate}%</StatValue>
        <StatLabel>Annotated</StatLabel>
      </StatCard>
      <StatCard>
        <StatValue>{statistics.training_completion_rate}%</StatValue>
        <StatLabel>Trained</StatLabel>
      </StatCard>

    </StatsContainer>
  );
}