import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Loader from '../components/Loader';
import { CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface RiskAnalysis {
  txnRisk: number;
  graphRisk: number;
  ipRisk: number;
  deviceRisk: number;
  finalRisk: number;
}

export default function TransactionPage() {
  const navigate = useNavigate();
  const [amount, setAmount] = useState('');
  const [deviceId, setDeviceId] = useState('');
  const [ipAddress, setIpAddress] = useState('');
  const [location, setLocation] = useState('');
  const [transactionType, setTransactionType] = useState('Transfer');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RiskAnalysis | null>(null);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  const generateRandomRisk = () => Math.random();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!amount || !deviceId || !ipAddress) {
      alert('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setResult(null);

    await new Promise((resolve) => setTimeout(resolve, 2000));

    const analysis: RiskAnalysis = {
      txnRisk: parseFloat((generateRandomRisk() * 0.6).toFixed(2)),
      graphRisk: parseFloat((generateRandomRisk() * 0.5).toFixed(2)),
      ipRisk: parseFloat((generateRandomRisk() * 0.8).toFixed(2)),
      deviceRisk: parseFloat((generateRandomRisk() * 0.4).toFixed(2)),
      finalRisk: 0,
    };

    analysis.finalRisk = parseFloat(
      ((analysis.txnRisk + analysis.graphRisk + analysis.ipRisk + analysis.deviceRisk) / 4).toFixed(2)
    );

    setResult(analysis);
    setLoading(false);
  };

  const getDecision = () => {
    if (!result) return null;

    if (result.finalRisk < 0.4) {
      return {
        status: 'APPROVED',
        color: 'green',
        icon: CheckCircle,
        message: 'Transaction has been approved and can proceed.',
      };
    } else if (result.finalRisk < 0.7) {
      return {
        status: 'STEP-UP AUTH',
        color: 'yellow',
        icon: AlertCircle,
        message: 'Additional authentication required before proceeding.',
      };
    } else {
      return {
        status: 'BLOCKED',
        color: 'red',
        icon: XCircle,
        message: 'Transaction flagged for review and has been blocked.',
      };
    }
  };

  const handleProceedTransaction = () => {
    const newTransaction = {
      id: `TXN-${Math.floor(Math.random() * 1000)}-${Math.floor(Math.random() * 10000)}`,
      amount: parseFloat(amount),
      deviceId,
      ipAddress,
      riskScore: result!.finalRisk,
      status: result!.finalRisk < 0.4 ? 'Approved' : result!.finalRisk < 0.7 ? 'Pending' : 'Flagged',
      timestamp: new Date().toISOString(),
    };

    const storedTransactions = localStorage.getItem('user_transactions');
    const transactions = storedTransactions ? JSON.parse(storedTransactions) : [];
    transactions.unshift(newTransaction);
    localStorage.setItem('user_transactions', JSON.stringify(transactions));

    setToastMessage('Transaction processed successfully!');
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);

    setTimeout(() => {
      navigate('/dashboard');
    }, 1500);
  };

  const decision = getDecision();

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar showDashboardLink showTransactionLink />

      {showToast && (
        <div className="fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50">
          {toastMessage}
        </div>
      )}

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">New Transaction</h1>
          <p className="text-gray-400">Submit transaction details for fraud analysis</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-6">Transaction Details</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Amount *
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="1000.00"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Device ID *
                </label>
                <input
                  type="text"
                  value={deviceId}
                  onChange={(e) => setDeviceId(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="DEV-1234-A"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  IP Address *
                </label>
                <input
                  type="text"
                  value={ipAddress}
                  onChange={(e) => setIpAddress(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="192.168.1.1"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Location (Optional)
                </label>
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  placeholder="New York, USA"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Transaction Type
                </label>
                <select
                  value={transactionType}
                  onChange={(e) => setTransactionType(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option>Transfer</option>
                  <option>Payment</option>
                  <option>Withdrawal</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {loading ? <Loader /> : 'Run Fraud Check'}
              </button>
            </form>
          </div>

          <div>
            {loading && (
              <div className="bg-gray-800 rounded-xl p-8 border border-gray-700 flex items-center justify-center h-full">
                <div className="text-center">
                  <Loader />
                  <p className="text-gray-400 mt-4">Processing fraud analysis...</p>
                </div>
              </div>
            )}

            {result && decision && (
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h2 className="text-xl font-bold text-white mb-6">Transaction Result</h2>

                <div className="space-y-4 mb-6">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Transaction Risk:</span>
                    <span className="text-white font-semibold">{result.txnRisk.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Graph Risk:</span>
                    <span className="text-white font-semibold">{result.graphRisk.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">IP Risk:</span>
                    <span className="text-white font-semibold">{result.ipRisk.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Device Risk:</span>
                    <span className="text-white font-semibold">{result.deviceRisk.toFixed(2)}</span>
                  </div>

                  <div className="border-t border-gray-700 pt-4 mt-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 font-semibold">Final Risk Score:</span>
                      <span className="text-white text-2xl font-bold">{result.finalRisk.toFixed(2)}</span>
                    </div>
                  </div>
                </div>

                <div className={`bg-${decision.color}-500/10 border border-${decision.color}-500 rounded-lg p-4 mb-4`}>
                  <div className="flex items-center space-x-3 mb-2">
                    <decision.icon className={`h-6 w-6 text-${decision.color}-500`} />
                    <span className={`text-${decision.color}-500 font-bold text-lg`}>
                      {decision.status}
                    </span>
                  </div>
                  <p className={`text-${decision.color}-400 text-sm`}>{decision.message}</p>
                </div>

                {result.finalRisk < 0.4 && (
                  <button
                    onClick={handleProceedTransaction}
                    className="w-full py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors"
                  >
                    Proceed Transaction
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
