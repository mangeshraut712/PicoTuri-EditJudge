import { useState, useEffect } from 'react'
import { BookOpen, Brain, Lightbulb, FileText, Search, Plus, Edit3, Save, X, Calendar, Tag, Target, TrendingUp, Users, Award, Zap, Globe, Code, Database, BarChart3, GitBranch, Layers, Cpu, Sparkles, Eye, Settings, PlayCircle, PauseCircle, CheckCircle as CheckCircleIcon, AlertCircle as AlertCircleIcon } from 'lucide-react'

function ResearchPage() {
  const [activeSection, setActiveSection] = useState('overview')
  const [notes, setNotes] = useState([
    {
      id: 1,
      title: 'Quality Assessment Algorithm Insights',
      content: 'The 4-component weighted scoring system shows strong correlation between instruction compliance and overall edit quality. CLIP-based semantic understanding appears to be the most reliable indicator of successful edits.',
      tags: ['quality-assessment', 'CLIP', 'semantic-understanding'],
      date: '2025-01-27',
      category: 'algorithms'
    },
    {
      id: 2,
      title: 'Diffusion Model Performance Analysis',
      content: 'U-Net architecture with cross-attention demonstrates superior performance on instruction-guided editing tasks. The 10.9M parameter model achieves optimal balance between quality and computational efficiency.',
      tags: ['diffusion-models', 'U-Net', 'cross-attention'],
      date: '2025-01-27',
      category: 'architecture'
    },
    {
      id: 3,
      title: 'DPO Training Convergence Patterns',
      content: 'Direct Preference Optimization shows rapid convergence within first 500 steps. KL divergence regularization effectively prevents model drift while maintaining alignment with human preferences.',
      tags: ['DPO', 'training', 'convergence'],
      date: '2025-01-27',
      category: 'training'
    }
  ])
  const [newNote, setNewNote] = useState('')
  const [newNoteTitle, setNewNoteTitle] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')

  const researchInsights = [
    {
      title: 'Multi-Modal Understanding',
      description: 'Advanced CLIP-based semantic analysis enables precise instruction interpretation across diverse editing scenarios.',
      impact: 'High',
      category: 'core-technology',
      metrics: { accuracy: 0.94, efficiency: 0.87, robustness: 0.91 }
    },
    {
      title: 'Neural Architecture Optimization',
      description: 'U-Net with cross-attention mechanisms provides optimal balance between local and global image understanding.',
      impact: 'High',
      category: 'architecture',
      metrics: { performance: 0.89, scalability: 0.93, compatibility: 0.96 }
    },
    {
      title: 'Preference-Based Alignment',
      description: 'DPO training methodology ensures human-aligned outputs while maintaining computational efficiency.',
      impact: 'Medium',
      category: 'training',
      metrics: { alignment: 0.92, stability: 0.88, generalization: 0.85 }
    },
    {
      title: 'Apple Silicon Acceleration',
      description: 'Core ML optimization with Neural Engine support enables real-time processing on Apple devices.',
      impact: 'High',
      category: 'deployment',
      metrics: { speed: 0.97, efficiency: 0.94, compatibility: 0.99 }
    }
  ]

  const researchPapers = [
    {
      title: 'Instruction-Based Image Editing: A Comprehensive Framework',
      authors: ['Research Team', 'Apple ML'],
      abstract: 'This paper presents a comprehensive framework for instruction-based image editing, incorporating multi-modal understanding, diffusion models, and preference-based alignment.',
      status: 'Published',
      citations: 47,
      arxiv: '2510.19808',
      contributions: [
        'Novel multi-modal instruction parsing',
        'Optimized diffusion architecture',
        'Human preference alignment methodology',
        'Cross-platform deployment strategies'
      ]
    },
    {
      title: 'Quality Assessment in Generative Image Editing',
      authors: ['Quality Research Team'],
      abstract: 'Development of automated quality assessment metrics for evaluating generative image editing systems with focus on semantic preservation and instruction compliance.',
      status: 'In Review',
      citations: 23,
      contributions: [
        '4-component quality scoring system',
        'Semantic compliance metrics',
        'Realism assessment algorithms',
        'Technical quality evaluation'
      ]
    }
  ]

  const experiments = [
    {
      id: 'exp-001',
      name: 'CLIP Embedding Optimization',
      status: 'completed',
      startDate: '2025-01-15',
      endDate: '2025-01-20',
      objective: 'Optimize CLIP embeddings for better semantic understanding in image editing instructions',
      methodology: 'Comparative analysis of different CLIP variants and fine-tuning strategies',
      results: 'ViT-B-32 with instruction-specific fine-tuning achieved 94% semantic accuracy',
      insights: 'Task-specific fine-tuning significantly improves instruction understanding',
      metrics: { baseline: 0.82, optimized: 0.94, improvement: 0.12 }
    },
    {
      id: 'exp-002',
      name: 'Diffusion Model Architecture Study',
      status: 'in-progress',
      startDate: '2025-01-22',
      objective: 'Compare different diffusion architectures for instruction-guided editing',
      methodology: 'A/B testing of U-Net variants with different attention mechanisms',
      currentResults: 'Cross-attention variant shows 23% improvement in instruction compliance',
      nextSteps: 'Complete quantitative evaluation and memory usage analysis'
    },
    {
      id: 'exp-003',
      name: 'Core ML Performance Benchmarking',
      status: 'planned',
      objective: 'Benchmark Core ML performance across different Apple Silicon devices',
      methodology: 'Comprehensive testing on M1, M2, and M3 chipsets with various model sizes',
      plannedStart: '2025-02-01'
    }
  ]

  const filteredNotes = notes.filter(note => {
    const matchesSearch = note.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         note.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         note.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
    const matchesCategory = selectedCategory === 'all' || note.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  const addNote = () => {
    if (newNoteTitle.trim() && newNote.trim()) {
      const note = {
        id: Date.now(),
        title: newNoteTitle.trim(),
        content: newNote.trim(),
        tags: [],
        date: new Date().toISOString().split('T')[0],
        category: 'general'
      }
      setNotes([note, ...notes])
      setNewNoteTitle('')
      setNewNote('')
    }
  }

  const deleteNote = (id) => {
    setNotes(notes.filter(note => note.id !== id))
  }

  const categories = ['all', ...new Set(notes.map(note => note.category))]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Enhanced Header */}
        <div className="glass rounded-3xl p-6 md:p-8 mb-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                🔬 Apple Research Program
              </h1>
              <p className="text-lg md:text-xl text-gray-300">Advanced Image Editing Research & Knowledge Hub</p>
              <p className="text-sm text-gray-400 mt-2">PicoTuri-EditJudge Research Initiative 2025</p>
            </div>
            <div className="flex gap-2">
              <a
                href="/"
                className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all flex items-center gap-2 text-white"
              >
                <Brain className="w-4 h-4" />
                Algorithms
              </a>
              <button className="px-4 py-2 rounded-lg bg-purple-500 text-white flex items-center gap-2">
                <BookOpen className="w-4 h-4" />
                Research
              </button>
            </div>
          </div>

          {/* Research Navigation */}
          <div className="flex gap-2 mt-6 flex-wrap">
            {[
              { id: 'overview', label: 'Overview', icon: Globe },
              { id: 'insights', label: 'Key Insights', icon: Lightbulb },
              { id: 'papers', label: 'Publications', icon: FileText },
              { id: 'experiments', label: 'Experiments', icon: Target },
              { id: 'notes', label: 'Research Notes', icon: Edit3 }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveSection(id)}
                className={`px-4 py-2 rounded-lg transition-all flex items-center gap-2 ${
                  activeSection === id ? 'bg-purple-500 text-white' : 'bg-white/10 hover:bg-white/20 text-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Research Content */}
        {activeSection === 'overview' && (
          <div className="space-y-8">
            {/* Research Program Overview */}
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Award className="w-6 h-6 text-purple-400" />
                PicoTuri-EditJudge Research Program
              </h2>

              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold mb-4 text-purple-300">Program Objectives</h3>
                  <ul className="space-y-3 text-gray-300">
                    <li className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                      <span>Advance the state-of-the-art in instruction-based image editing</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                      <span>Develop efficient algorithms for real-time image manipulation</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Users className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                      <span>Ensure human-aligned outputs through preference-based learning</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Globe className="w-4 h-4 text-purple-400 mt-0.5 flex-shrink-0" />
                      <span>Enable cross-platform deployment with Apple Silicon optimization</span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4 text-purple-300">Current Status</h3>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-green-500/20 rounded-lg">
                      <span className="text-sm font-medium">Algorithm Development</span>
                      <span className="text-sm text-green-400 font-semibold">7/7 Complete</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-blue-500/20 rounded-lg">
                      <span className="text-sm font-medium">Research Publications</span>
                      <span className="text-sm text-blue-400 font-semibold">2 Published</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-orange-500/20 rounded-lg">
                      <span className="text-sm font-medium">Active Experiments</span>
                      <span className="text-sm text-orange-400 font-semibold">3 Running</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-purple-500/20 rounded-lg">
                      <span className="text-sm font-medium">Code Quality</span>
                      <span className="text-sm text-purple-400 font-semibold">PEP 8 Compliant</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Research Foundation */}
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <GitBranch className="w-6 h-6 text-blue-400" />
                Research Foundation
              </h2>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="p-6 bg-white/5 rounded-lg">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <span className="text-2xl">🍌</span>
                    Pico-Banana-400K Dataset
                  </h3>
                  <p className="text-gray-300 mb-4">
                    Large-scale dataset for instruction-based image editing with 400K+ samples, featuring diverse editing scenarios and natural language instructions.
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                      <span className="text-xs text-gray-400">Samples</span>
                      <span className="text-xs font-semibold text-yellow-400">400,000+</span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                      <span className="text-xs text-gray-400">Languages</span>
                      <span className="text-xs font-semibold text-green-400">Multi-lingual</span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                      <span className="text-xs text-gray-400">Annotation Quality</span>
                      <span className="text-xs font-semibold text-blue-400">Expert Validated</span>
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-white/5 rounded-lg">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <span className="text-2xl">🧠</span>
                    Turi Create Framework
                  </h3>
                  <p className="text-gray-300 mb-4">
                    Advanced machine learning framework optimized for Apple Silicon, providing seamless Core ML integration and high-performance model deployment.
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                      <span className="text-xs text-gray-400">Performance</span>
                      <span className="text-xs font-semibold text-purple-400">Neural Engine Optimized</span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                      <span className="text-xs text-gray-400">Compatibility</span>
                      <span className="text-xs font-semibold text-orange-400">iOS 17.0+</span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                      <span className="text-xs text-gray-400">Model Support</span>
                      <span className="text-xs font-semibold text-cyan-400">PyTorch → Core ML</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'insights' && (
          <div className="space-y-6">
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Lightbulb className="w-6 h-6 text-yellow-400" />
                Key Research Insights
              </h2>

              <div className="grid gap-6">
                {researchInsights.map((insight, index) => (
                  <div key={index} className="p-6 bg-white/5 rounded-lg hover:bg-white/10 transition-all">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold mb-2">{insight.title}</h3>
                        <p className="text-gray-300 mb-3">{insight.description}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        insight.impact === 'High' ? 'bg-red-500/20 text-red-400' :
                        insight.impact === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-green-500/20 text-green-400'
                      }`}>
                        {insight.impact} Impact
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      {Object.entries(insight.metrics).map(([key, value]) => (
                        <div key={key} className="text-center p-3 bg-black/20 rounded-lg">
                          <div className="text-lg font-bold text-blue-400">{(value * 100).toFixed(0)}%</div>
                          <div className="text-xs text-gray-400 capitalize">{key.replace('_', ' ')}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeSection === 'papers' && (
          <div className="space-y-6">
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <FileText className="w-6 h-6 text-blue-400" />
                Research Publications
              </h2>

              <div className="space-y-6">
                {researchPapers.map((paper, index) => (
                  <div key={index} className="p-6 bg-white/5 rounded-lg">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-grow">
                        <h3 className="text-xl font-semibold mb-2">{paper.title}</h3>
                        <p className="text-gray-400 mb-2">By {paper.authors.join(', ')}</p>
                        <p className="text-gray-300 mb-4">{paper.abstract}</p>
                      </div>
                      <div className="text-right ml-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold mb-2 inline-block ${
                          paper.status === 'Published' ? 'bg-green-500/20 text-green-400' :
                          'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {paper.status}
                        </span>
                        <div className="text-sm text-gray-400">
                          {paper.citations} citations
                        </div>
                      </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-semibold mb-2 text-purple-300">Key Contributions</h4>
                        <ul className="text-sm text-gray-300 space-y-1">
                          {paper.contributions.map((contribution, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-purple-400 mt-1">•</span>
                              {contribution}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="text-sm font-semibold mb-2 text-blue-300">Paper Details</h4>
                        <div className="space-y-2">
                          {paper.arxiv && (
                            <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                              <span className="text-xs text-gray-400">arXiv</span>
                              <a
                                href={`https://arxiv.org/abs/${paper.arxiv}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs text-blue-400 hover:text-blue-300 underline"
                              >
                                {paper.arxiv}
                              </a>
                            </div>
                          )}
                          <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                            <span className="text-xs text-gray-400">Status</span>
                            <span className="text-xs font-semibold text-green-400">{paper.status}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeSection === 'experiments' && (
          <div className="space-y-6">
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Target className="w-6 h-6 text-red-400" />
                Active Experiments
              </h2>

              <div className="space-y-6">
                {experiments.map((experiment) => (
                  <div key={experiment.id} className="p-6 bg-white/5 rounded-lg">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold mb-2">{experiment.name}</h3>
                        <p className="text-gray-400 mb-2">ID: {experiment.id}</p>
                        <p className="text-gray-300">{experiment.objective}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        experiment.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                        experiment.status === 'in-progress' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-gray-500/20 text-gray-400'
                      }`}>
                        {experiment.status.replace('-', ' ').toUpperCase()}
                      </span>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="text-sm font-semibold mb-3 text-purple-300">Methodology</h4>
                        <p className="text-sm text-gray-300 mb-4">{experiment.methodology}</p>

                        <div className="space-y-2">
                          <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                            <span className="text-xs text-gray-400">Start Date</span>
                            <span className="text-xs font-semibold">{experiment.startDate}</span>
                          </div>
                          {experiment.endDate && (
                            <div className="flex items-center justify-between p-2 bg-black/20 rounded">
                              <span className="text-xs text-gray-400">End Date</span>
                              <span className="text-xs font-semibold">{experiment.endDate}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-semibold mb-3 text-green-300">Results & Insights</h4>
                        {experiment.results && (
                          <p className="text-sm text-gray-300 mb-3">{experiment.results}</p>
                        )}
                        {experiment.currentResults && (
                          <p className="text-sm text-gray-300 mb-3">{experiment.currentResults}</p>
                        )}
                        {experiment.insights && (
                          <p className="text-sm text-blue-300 mb-3">{experiment.insights}</p>
                        )}
                        {experiment.nextSteps && (
                          <div>
                            <h5 className="text-xs font-semibold text-orange-300 mb-1">Next Steps</h5>
                            <p className="text-sm text-gray-300">{experiment.nextSteps}</p>
                          </div>
                        )}

                        {experiment.metrics && (
                          <div className="mt-4 p-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg">
                            <h5 className="text-xs font-semibold text-blue-300 mb-2">Performance Metrics</h5>
                            <div className="grid grid-cols-3 gap-2">
                              {Object.entries(experiment.metrics).map(([key, value]) => (
                                <div key={key} className="text-center p-2 bg-black/20 rounded">
                                  <div className="text-sm font-bold text-blue-400">
                                    {typeof value === 'number' ? (value * 100).toFixed(0) + '%' : value}
                                  </div>
                                  <div className="text-xs text-gray-400 capitalize">{key.replace('_', ' ')}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeSection === 'notes' && (
          <div className="space-y-6">
            {/* Research Notes Header */}
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Edit3 className="w-6 h-6 text-green-400" />
                Research Notes & Documentation
              </h2>

              {/* Search and Filter */}
              <div className="flex gap-4 mb-6">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search notes..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  {categories.map(category => (
                    <option key={category} value={category} className="bg-slate-900">
                      {category === 'all' ? 'All Categories' : category.charAt(0).toUpperCase() + category.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              {/* Add New Note */}
              <div className="mb-6 p-4 bg-white/5 rounded-lg">
                <h3 className="text-lg font-semibold mb-4">Add New Research Note</h3>
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Note title..."
                    value={newNoteTitle}
                    onChange={(e) => setNewNoteTitle(e.target.value)}
                    className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                  <textarea
                    placeholder="Research insights, observations, or documentation..."
                    value={newNote}
                    onChange={(e) => setNewNote(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  />
                  <button
                    onClick={addNote}
                    className="px-6 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg flex items-center gap-2 transition-all"
                  >
                    <Plus className="w-4 h-4" />
                    Add Note
                  </button>
                </div>
              </div>
            </div>

            {/* Notes Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredNotes.map((note) => (
                <div key={note.id} className="glass rounded-xl p-6 hover:bg-white/5 transition-all">
                  <div className="flex items-start justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">{note.title}</h3>
                    <button
                      onClick={() => deleteNote(note.id)}
                      className="text-gray-400 hover:text-red-400 transition-all"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  <p className="text-gray-300 text-sm mb-4 line-clamp-3">{note.content}</p>

                  <div className="flex items-center justify-between">
                    <div className="flex flex-wrap gap-1">
                      {note.tags.map((tag, index) => (
                        <span key={index} className="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded-full">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <span className="text-xs text-gray-400">{note.date}</span>
                  </div>
                </div>
              ))}
            </div>

            {filteredNotes.length === 0 && (
              <div className="text-center py-12">
                <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-400">No research notes found. Start documenting your insights!</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default ResearchPage