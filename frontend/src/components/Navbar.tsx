import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Shield, LogOut } from 'lucide-react';

interface NavbarProps {
  showDashboardLink?: boolean;
  showTransactionLink?: boolean;
  isAdmin?: boolean;
}

export default function Navbar({ showDashboardLink, showTransactionLink, isAdmin }: NavbarProps) {
  const { logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to={isAdmin ? '/admin' : '/dashboard'} className="flex items-center space-x-2">
            <Shield className="h-8 w-8 text-blue-500" />
            <span className="text-xl font-bold text-white">FinShield</span>
          </Link>

          <div className="flex items-center space-x-6">
            {showDashboardLink && (
              <Link
                to={isAdmin ? '/admin' : '/dashboard'}
                className="text-gray-300 hover:text-white transition-colors"
              >
                {isAdmin ? 'Admin Dashboard' : 'Dashboard'}
              </Link>
            )}
            {showTransactionLink && !isAdmin && (
              <Link
                to="/transaction"
                className="text-gray-300 hover:text-white transition-colors"
              >
                New Transaction
              </Link>
            )}
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 text-gray-300 hover:text-white transition-colors"
            >
              <LogOut className="h-5 w-5" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
