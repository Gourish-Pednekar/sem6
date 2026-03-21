import { Link } from 'react-router-dom';
import { Shield, Activity, Network, Eye } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-blue-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-16 pt-12">
          <div className="flex justify-center mb-6">
            <Shield className="h-20 w-20 text-blue-500" />
          </div>
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            FinShield
          </h1>
          <h2 className="text-3xl md:text-4xl font-semibold text-blue-400 mb-6">
            Real-Time Multi-Layer Fraud Detection
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-10">
            AI-powered transaction monitoring using behavioral, graph and entity intelligence.
          </p>
          <div className="flex justify-center space-x-4">
            <Link
              to="/login"
              className="px-8 py-4 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition-colors shadow-lg"
            >
              Login
            </Link>
            <Link
              to="/register"
              className="px-8 py-4 bg-gray-800 text-white rounded-xl font-semibold hover:bg-gray-700 transition-colors shadow-lg border border-gray-700"
            >
              Register
            </Link>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-gray-800 rounded-xl p-8 hover:bg-gray-750 transition-all hover:shadow-xl">
            <Activity className="h-12 w-12 text-blue-500 mb-4" />
            <h3 className="text-2xl font-bold text-white mb-3">
              Transaction Risk Engine
            </h3>
            <p className="text-gray-300">
              Advanced machine learning algorithms analyze transaction patterns in real-time to detect anomalies and prevent fraud before it happens.
            </p>
          </div>

          <div className="bg-gray-800 rounded-xl p-8 hover:bg-gray-750 transition-all hover:shadow-xl">
            <Network className="h-12 w-12 text-blue-500 mb-4" />
            <h3 className="text-2xl font-bold text-white mb-3">
              Graph-Based Fraud Detection
            </h3>
            <p className="text-gray-300">
              Identify complex fraud networks by analyzing relationships between users, devices, and IP addresses across multiple transactions.
            </p>
          </div>

          <div className="bg-gray-800 rounded-xl p-8 hover:bg-gray-750 transition-all hover:shadow-xl">
            <Eye className="h-12 w-12 text-blue-500 mb-4" />
            <h3 className="text-2xl font-bold text-white mb-3">
              Real-Time Admin Monitoring
            </h3>
            <p className="text-gray-300">
              Comprehensive dashboard for fraud analysts to monitor flagged transactions, manage risk entities, and take immediate action.
            </p>
          </div>
        </div>

        <footer className="text-center text-gray-400 pt-12 border-t border-gray-800">
          <div className="flex justify-center space-x-6 mb-4">
            <a href="#" className="hover:text-white transition-colors">About</a>
            <a href="#" className="hover:text-white transition-colors">Privacy</a>
            <a href="#" className="hover:text-white transition-colors">Terms</a>
            <a href="#" className="hover:text-white transition-colors">Contact</a>
          </div>
          <p>&copy; 2024 FraudGuard AI. All rights reserved.</p>
        </footer>
      </div>
    </div>
  );
}
