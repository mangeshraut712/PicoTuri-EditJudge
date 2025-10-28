import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Brain, Activity, BookOpen, Settings, BarChart3, Zap, Menu, X } from 'lucide-react'

const Navigation = ({ isOpen, onToggle }) => {
  const location = useLocation()

  const navItems = [
    {
      name: 'Algorithms',
      path: '/',
      icon: Brain,
      description: 'Test and analyze all algorithms',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      name: 'Performance',
      path: '/performance',
      icon: Activity,
      description: 'Real-time monitoring dashboard',
      color: 'from-green-500 to-emerald-500'
    },
    {
      name: 'Research',
      path: '/research',
      icon: BookOpen,
      description: 'Academic papers and documentation',
      color: 'from-purple-500 to-pink-500'
    }
  ]

  const isActive = (path) => location.pathname === path

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={onToggle}
        className="fixed top-4 left-4 z-50 lg:hidden p-3 rounded-xl bg-black/20 backdrop-blur-sm border border-white/10 text-white hover:bg-black/30 transition-all"
      >
        {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Sidebar */}
      <div className={`fixed left-0 top-0 h-full w-80 bg-black/20 backdrop-blur-xl border-r border-white/10 transform transition-transform duration-300 z-40 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      } lg:translate-x-0`}>
        <div className="p-6 h-full overflow-y-auto">
          {/* Logo and Title */}
          <div className="mb-8">
            <Link to="/" className="block group">
              <div className="flex items-center gap-3 mb-2">
                <div className="p-2 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 group-hover:scale-110 transition-transform duration-300">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                    PicoTuri
                  </h1>
                  <p className="text-sm text-gray-400">EditJudge 2025</p>
                </div>
              </div>
              <p className="text-sm text-gray-300 group-hover:text-white transition-colors">
                Advanced AI Quality Assessment Platform
              </p>
            </Link>
          </div>

          {/* Navigation Items */}
          <nav className="space-y-3 mb-8">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => onToggle(false)}
                  className={`block p-4 rounded-xl transition-all duration-300 group ${
                    isActive(item.path)
                      ? 'bg-gradient-to-r ' + item.color + ' bg-opacity-20 border border-white/20'
                      : 'hover:bg-white/5 border border-transparent hover:border-white/10'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      isActive(item.path)
                        ? 'bg-gradient-to-r ' + item.color
                        : 'bg-white/10 group-hover:bg-white/20'
                    } transition-all duration-300`}>
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className={`font-semibold mb-1 ${
                        isActive(item.path) ? 'text-white' : 'text-gray-300 group-hover:text-white'
                      } transition-colors`}>
                        {item.name}
                      </h3>
                      <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors">
                        {item.description}
                      </p>
                    </div>
                  </div>
                </Link>
              )
            })}
          </nav>

          {/* System Status */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
              System Status
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-green-400">Frontend</span>
                </div>
                <span className="text-xs text-green-400">Active</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-blue-400">Backend</span>
                </div>
                <span className="text-xs text-blue-400">Port 5001</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-purple-400">Algorithms</span>
                </div>
                <span className="text-xs text-purple-400">7 Available</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => onToggle(false)}
        />
      )}
    </>
  )
}

export default Navigation
