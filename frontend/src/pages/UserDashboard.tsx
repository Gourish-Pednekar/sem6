import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Navbar from '../components/Navbar';
import RiskBadge from '../components/RiskBadge';
import { TrendingUp, CheckCircle, AlertTriangle, Activity } from 'lucide-react';

interface Transaction {
  id: string;
  amount: number;
  deviceId: string;
  ipAddress: string;
  riskScore: number;
  status: 'Approved' | 'Flagged' | 'Pending';
  timestamp: string;
}

export default function UserDashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [transactions, setTransactions] = useState<Transaction[]>([]);

  useEffect(() => {
    const mockTransactions: Transaction[] = [
      {
        id: 'TXN-001-8472',
        amount: 1250.00,
        deviceId: 'DEV-8472-A',
        ipAddress: '192.168.1.45',
        riskScore: 0.25,
        status: 'Approved',
        timestamp: '2024-01-15 14:32:10',
      },
      {
        id: 'TXN-002-9183',
        amount: 3400.00,
        deviceId: 'DEV-9183-B',
        ipAddress: '10.0.0.125',
        riskScore: 0.68,
        status: 'Flagged',
        timestamp: '2024-01-15 12:15:42',
      },
      {
        id: 'TXN-003-5629',
        amount: 850.50,
        deviceId: 'DEV-5629-C',
        ipAddress: '172.16.0.88',
        riskScore: 0.32,
        status: 'Approved',
        timestamp: '2024-01-14 16:45:22',
      },
      {
        id: 'TXN-004-7341',
        amount: 5200.00,
        deviceId: 'DEV-7341-D',
        ipAddress: '192.168.2.101',
        riskScore: 0.55,
        status: 'Pending',
        timestamp: '2024-01-14 10:20:15',
      },
      {
        id: 'TXN-005-2984',
        amount: 475.25,
        deviceId: 'DEV-2984-E',
        ipAddress: '10.1.1.50',
        riskScore: 0.18,
        status: 'Approved',
        timestamp: '2024-01-13 09:10:33',
      },
    ];

    const storedTransactions = localStorage.getItem('user_transactions');
    if (storedTransactions) {
      setTransactions(JSON.parse(storedTransactions));
    } else {
      setTransactions(mockTransactions);
      localStorage.setItem('user_transactions', JSON.stringify(mockTransactions));
    }
  }, []);

  const stats = {
    total: transactions.length,
    approved: transactions.filter((t) => t.status === 'Approved').length,
    flagged: transactions.filter((t) => t.status === 'Flagged').length,
    avgRisk: transactions.reduce((sum, t) => sum + t.riskScore, 0) / transactions.length || 0,
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar showDashboardLink showTransactionLink />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            Welcome back, {user?.name}!
          </h1>
          <p className="text-gray-400">Here's your transaction overview</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <TrendingUp className="h-8 w-8 text-blue-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">Total Transactions</p>
            <p className="text-3xl font-bold text-white">{stats.total}</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">Approved</p>
            <p className="text-3xl font-bold text-white">{stats.approved}</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">Flagged</p>
            <p className="text-3xl font-bold text-white">{stats.flagged}</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Activity className="h-8 w-8 text-yellow-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">Avg Risk Score</p>
            <p className="text-3xl font-bold text-white">{stats.avgRisk.toFixed(2)}</p>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-white">Recent Transactions</h2>
            <button
              onClick={() => navigate('/transaction')}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Initiate New Transaction
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Transaction ID</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Amount</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Device ID</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">IP Address</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Risk Score</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((transaction) => (
                  <tr key={transaction.id} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-4 px-4 text-white font-mono text-sm">{transaction.id}</td>
                    <td className="py-4 px-4 text-white font-semibold">${transaction.amount.toFixed(2)}</td>
                    <td className="py-4 px-4 text-gray-300 font-mono text-sm">{transaction.deviceId}</td>
                    <td className="py-4 px-4 text-gray-300 font-mono text-sm">{transaction.ipAddress}</td>
                    <td className="py-4 px-4">
                      <RiskBadge score={transaction.riskScore} />
                    </td>
                    <td className="py-4 px-4">
                      <span
                        className={`font-semibold ${
                          transaction.status === 'Approved'
                            ? 'text-green-400'
                            : transaction.status === 'Flagged'
                            ? 'text-red-400'
                            : 'text-yellow-400'
                        }`}
                      >
                        {transaction.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
