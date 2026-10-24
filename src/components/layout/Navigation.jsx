import { useEffect, useRef } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Brain, Activity, BookOpen, Menu, X } from 'lucide-react'
import { trapTabKey } from '../../utils/focusTrap'

const Navigation = ({ isOpen, onToggle }) => {
  const location = useLocation()
  const menuRef = useRef(null)
  const toggleRef = useRef(null)

  useEffect(() => {
    if (!isOpen) return undefined
    const menu = menuRef.current
    const onKey = (event) => {
      if (event.key === 'Escape') {
        if (event.defaultPrevented) return
        onToggle(false)
        return
      }
      trapTabKey(event, menu)
    }
    document.addEventListener('keydown', onKey)
    toggleRef.current?.focus()
    return () => document.removeEventListener('keydown', onKey)
  }, [isOpen, onToggle])

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
      description: 'Performance samples and demo metrics',
      color: 'from-green-500 to-emerald-500'
    },
    {
      name: 'Research',
      path: '/research',
      icon: BookOpen,
      description: 'Sample bibliography',
      color: 'from-purple-500 to-pink-500'
    }
  ]

  const isActive = (path) => location.pathname === path

  return (
    <div ref={menuRef}>
      {/* Mobile Menu Button */}
      <button
        ref={toggleRef}
        type="button"
        onClick={() => onToggle(!isOpen)}
        aria-label={isOpen ? 'Close navigation menu' : 'Open navigation menu'}
        aria-expanded={isOpen}
        aria-controls="primary-navigation"
        className="fixed top-4 left-4 z-50 lg:hidden p-3 rounded-xl bg-black/20 backdrop-blur-sm border border-white/10 text-white hover:bg-black/30 transition-all"
      >
        {isOpen ? <X className="w-6 h-6" aria-hidden="true" /> : <Menu className="w-6 h-6" aria-hidden="true" />}
      </button>

      {/* Sidebar */}
      <div id="primary-navigation" className={`fixed left-0 top-0 h-full w-80 bg-black/20 backdrop-blur-xl border-r border-white/10 transform transition-transform duration-300 z-40 ${
        isOpen ? 'visible translate-x-0' : 'invisible -translate-x-full'
      } lg:visible lg:translate-x-0`}>
        <div className="p-6 h-full overflow-y-auto">
          {/* Logo and Title */}
          <div className="mb-8">
            <Link to="/" onClick={() => onToggle(false)} className="block group">
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
          <nav aria-label="Primary" className="space-y-3 mb-8">
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  aria-current={isActive(item.path) ? 'page' : undefined}
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

          {/* Demo contents. These are not probed from a health check. */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-1">
              Demo contents
            </h3>
            <p className="text-xs text-gray-500 mb-3">Not a live health check.</p>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                <span className="text-sm text-green-400">Frontend</span>
                <span className="text-xs text-green-400">This page</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <span className="text-sm text-blue-400">API</span>
                <span className="text-xs text-blue-400">Optional local server</span>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <span className="text-sm text-purple-400">Algorithms</span>
                <span className="text-xs text-purple-400">7 sample tests</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isOpen && (
        <button
          type="button"
          tabIndex={-1}
          aria-label="Dismiss navigation menu"
          className="fixed inset-0 z-30 border-0 bg-black/50 p-0 lg:hidden"
          onClick={() => onToggle(false)}
        />
      )}
    </div>
  )
}

export default Navigation
