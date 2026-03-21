import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import RiskBadge from '../components/RiskBadge';
import { Users, AlertTriangle, Wifi, Smartphone } from 'lucide-react';

interface FlaggedTransaction {
  id: string;
  user: string;
  amount: number;
  ip: string;
  device: string;
  riskScore: number;
}

interface RiskyEntity {
  id: string;
  fraudRate: string;
  totalTransactions?: number;
  uniqueUsers?: number;
}

export default function AdminDashboard() {
  const [flaggedTransactions, setFlaggedTransactions] = useState<FlaggedTransaction[]>([]);
  const [riskyIPs, setRiskyIPs] = useState<RiskyEntity[]>([]);
  const [riskyDevices, setRiskyDevices] = useState<RiskyEntity[]>([]);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  useEffect(() => {
    const initialFlagged: FlaggedTransaction[] = [
      {
        id: 'TXN-8472-F',
        user: 'john.doe@email.com',
        amount: 4500,
        ip: '45.89.123.45',
        device: 'DEV-8472-X',
        riskScore: 0.82,
      },
      {
        id: 'TXN-9183-F',
        user: 'jane.smith@email.com',
        amount: 7200,
        ip: '192.168.50.101',
        device: 'DEV-9183-Y',
        riskScore: 0.75,
      },
    ];

    const initialIPs: RiskyEntity[] = [
      { id: '45.89.123.45', fraudRate: '78%', totalTransactions: 145 },
      { id: '198.51.100.88', fraudRate: '65%', totalTransactions: 89 },
      { id: '203.0.113.50', fraudRate: '52%', totalTransactions: 67 },
    ];

    const initialDevices: RiskyEntity[] = [
      { id: 'DEV-8472-X', fraudRate: '82%', uniqueUsers: 23 },
      { id: 'DEV-5629-Z', fraudRate: '71%', uniqueUsers: 18 },
      { id: 'DEV-3341-W', fraudRate: '58%', uniqueUsers: 12 },
    ];

    setFlaggedTransactions(initialFlagged);
    setRiskyIPs(initialIPs);
    setRiskyDevices(initialDevices);

    const interval = setInterval(() => {
      const shouldAdd = Math.random() > 0.7;
      if (shouldAdd) {
        const newTransaction: FlaggedTransaction = {
          id: `TXN-${Math.floor(Math.random() * 10000)}-F`,
          user: `user${Math.floor(Math.random() * 1000)}@email.com`,
          amount: Math.floor(Math.random() * 10000) + 1000,
          ip: `${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`,
          device: `DEV-${Math.floor(Math.random() * 10000)}-${String.fromCharCode(65 + Math.floor(Math.random() * 26))}`,
          riskScore: parseFloat((0.7 + Math.random() * 0.3).toFixed(2)),
        };
        setFlaggedTransactions((prev) => [newTransaction, ...prev].slice(0, 10));
      }
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const showNotification = (message: string) => {
    setToastMessage(message);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  const handleApprove = (id: string) => {
    setFlaggedTransactions((prev) => prev.filter((t) => t.id !== id));
    showNotification('Transaction approved');
  };

  const handleBlockUser = (user: string) => {
    showNotification(`User ${user} has been blocked`);
  };

  const handleFreezeAccount = (user: string) => {
    showNotification(`Account ${user} has been frozen`);
  };

  const handleWhitelistIP = (ip: string) => {
    setRiskyIPs((prev) => prev.filter((i) => i.id !== ip));
    showNotification(`IP ${ip} has been whitelisted`);
  };

  const handleBlockDevice = (device: string) => {
    setRiskyDevices((prev) => prev.filter((d) => d.id !== device));
    showNotification(`Device ${device} has been blocked`);
  };

  const stats = {
    totalUsers: 1247,
    flaggedTransactions: flaggedTransactions.length,
    highRiskIPs: riskyIPs.length,
    highRiskDevices: riskyDevices.length,
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar showDashboardLink isAdmin />

      {showToast && (
        <div className="fixed top-4 right-4 bg-blue-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-fade-in">
          {toastMessage}
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Admin Dashboard</h1>
          <p className="text-gray-400">Monitor and manage fraud detection system</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Users className="h-8 w-8 text-blue-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">Total Users</p>
            <p className="text-3xl font-bold text-white">{stats.totalUsers}</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">Flagged Transactions</p>
            <p className="text-3xl font-bold text-white">{stats.flaggedTransactions}</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Wifi className="h-8 w-8 text-yellow-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">High Risk IPs</p>
            <p className="text-3xl font-bold text-white">{stats.highRiskIPs}</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <Smartphone className="h-8 w-8 text-orange-500" />
            </div>
            <p className="text-gray-400 text-sm mb-1">High Risk Devices</p>
            <p className="text-3xl font-bold text-white">{stats.highRiskDevices}</p>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Flagged Transactions</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">User</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Amount</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">IP</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Device</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Risk Score</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {flaggedTransactions.map((transaction) => (
                  <tr key={transaction.id} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-4 px-4 text-white">{transaction.user}</td>
                    <td className="py-4 px-4 text-white font-semibold">
                      ${transaction.amount.toFixed(2)}
                    </td>
                    <td className="py-4 px-4 text-gray-300 font-mono text-sm">{transaction.ip}</td>
                    <td className="py-4 px-4 text-gray-300 font-mono text-sm">{transaction.device}</td>
                    <td className="py-4 px-4">
                      <RiskBadge score={transaction.riskScore} />
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleApprove(transaction.id)}
                          className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 transition-colors"
                        >
                          Approve
                        </button>
                        <button
                          onClick={() => handleBlockUser(transaction.user)}
                          className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 transition-colors"
                        >
                          Block User
                        </button>
                        <button
                          onClick={() => handleFreezeAccount(transaction.user)}
                          className="px-3 py-1 bg-yellow-600 text-white text-sm rounded hover:bg-yellow-700 transition-colors"
                        >
                          Freeze
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-6">Risky IPs</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">IP Address</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Fraud Rate</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Transactions</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {riskyIPs.map((ip) => (
                    <tr key={ip.id} className="border-b border-gray-700 hover:bg-gray-750">
                      <td className="py-4 px-4 text-white font-mono text-sm">{ip.id}</td>
                      <td className="py-4 px-4 text-red-400 font-semibold">{ip.fraudRate}</td>
                      <td className="py-4 px-4 text-gray-300">{ip.totalTransactions}</td>
                      <td className="py-4 px-4">
                        <button
                          onClick={() => handleWhitelistIP(ip.id)}
                          className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                        >
                          Whitelist
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-6">Risky Devices</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Device ID</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Fraud Rate</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Users</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {riskyDevices.map((device) => (
                    <tr key={device.id} className="border-b border-gray-700 hover:bg-gray-750">
                      <td className="py-4 px-4 text-white font-mono text-sm">{device.id}</td>
                      <td className="py-4 px-4 text-red-400 font-semibold">{device.fraudRate}</td>
                      <td className="py-4 px-4 text-gray-300">{device.uniqueUsers}</td>
                      <td className="py-4 px-4">
                        <button
                          onClick={() => handleBlockDevice(device.id)}
                          className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 transition-colors"
                        >
                          Block Device
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
