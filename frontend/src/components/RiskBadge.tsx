interface RiskBadgeProps {
  score: number;
}

export default function RiskBadge({ score }: RiskBadgeProps) {
  const getRiskLevel = () => {
    if (score < 0.4) return { label: 'Low', color: 'bg-green-500' };
    if (score < 0.7) return { label: 'Medium', color: 'bg-yellow-500' };
    return { label: 'High', color: 'bg-red-500' };
  };

  const risk = getRiskLevel();

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-semibold text-white ${risk.color}`}>
      {risk.label}
    </span>
  );
}
