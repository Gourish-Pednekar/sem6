import RiskBadge from './RiskBadge';

interface Transaction {
  id: string;
  amount: number;
  deviceId: string;
  ipAddress: string;
  riskScore: number;
  status: 'Approved' | 'Flagged' | 'Pending';
}

interface TransactionCardProps {
  transaction: Transaction;
}

export default function TransactionCard({ transaction }: TransactionCardProps) {
  const getStatusColor = () => {
    switch (transaction.status) {
      case 'Approved':
        return 'text-green-400';
      case 'Flagged':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6 hover:bg-gray-750 transition-colors">
      <div className="flex justify-between items-start mb-4">
        <div>
          <p className="text-gray-400 text-sm">Transaction ID</p>
          <p className="text-white font-mono">{transaction.id}</p>
        </div>
        <RiskBadge score={transaction.riskScore} />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-gray-400 text-sm">Amount</p>
          <p className="text-white font-semibold">${transaction.amount.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-gray-400 text-sm">Status</p>
          <p className={`font-semibold ${getStatusColor()}`}>{transaction.status}</p>
        </div>
      </div>

      <div className="space-y-2">
        <div>
          <p className="text-gray-400 text-sm">Device ID</p>
          <p className="text-white text-sm font-mono">{transaction.deviceId}</p>
        </div>
        <div>
          <p className="text-gray-400 text-sm">IP Address</p>
          <p className="text-white text-sm font-mono">{transaction.ipAddress}</p>
        </div>
      </div>
    </div>
  );
}
