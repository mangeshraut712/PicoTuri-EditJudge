import React, { useState } from 'react'
import { BookOpen, FileText, Users, Award, TrendingUp, Brain, Code, BarChart3, ExternalLink, Search, Filter, Calendar } from 'lucide-react'

const ResearchPage = () => {
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  // Mock research data - academic papers, experiments, citations
  const researchPapers = [
    {
      title: "Multi-Modal Image Editing Quality Assessment Using CLIP and ResNet50",
      authors: ["Dr. Sarah Chen", "Dr. Michael Johnson", "Prof. Lisa Rodriguez"],
      journal: "IEEE Conference on Computer Vision and Pattern Recognition",
      year: 2024,
      citations: 247,
      abstract: "This paper introduces a novel framework for assessing image editing quality using both semantic understanding and technical fidelity metrics. We combine CLIP embeddings for semantic content analysis with ResNet50 for low-level image quality assessment.",
      category: "assessment",
      keywords: ["CLIP", "ResNet", "Quality Assessment", "Image Editing"],
      doi: "10.1109/CVPR57869.2024.01234",
      status: "published"
    },
    {
      title: "Diffusion Models for Instruction-Based Image Editing: A Comparative Study",
      authors: ["Dr. Alex Kumar", "Dr. Maria Garcia", "Prof. David Brown"],
      journal: "ACM Transactions on Graphics",
      year: 2024,
      citations: 189,
      abstract: "We present a comprehensive comparison of diffusion-based approaches for textual instruction-guided image editing. Our evaluation covers 7 major architectures and provides empirical evidence for optimal hyperparameter selection.",
      category: "generation",
      keywords: ["Diffusion Models", "Text-to-Image", "Instruction following"],
      doi: "10.1145/3627678.3626923",
      status: "published"
    },
    {
      title: "DPO Training: Preference-Based Alignment for Image Editing Models",
      authors: ["Dr. James Wilson", "Dr. Anna Petrov", "Prof. Robert Davis"],
      journal: "Machine Learning Research",
      year: 2024,
      citations: 134,
      abstract: "This work explores Direct Preference Optimization for aligning image editing models with human preferences. We demonstrate significant improvements in instruction compliance and user satisfaction through preference-based training.",
      category: "training",
      keywords: ["DPO", "Reinforcement Learning", "Preference Learning"],
      doi: "10.48550/arXiv.2407.12345",
      status: "preprint"
    },
    {
      title: "Conversation-Based Image Editing: Multi-Turn Instructions and Context Preservation",
      authors: ["Dr. Emily Zhang", "Dr. Thomas Anderson", "Prof. Susan Miller"],
      journal: "International Conference on Artificial Intelligence",
      year: 2024,
      citations: 96,
      abstract: "We introduce a novel multi-turn image editing paradigm where users can provide sequential editing instructions while preserving contextual relationships between edits.",
      category: "interaction",
      keywords: ["Conversational AI", "Multi-turn", "Context Preservation"],
      doi: "10.1609/aaai.v38i12.28984",
      status: "published"
    },
    {
      title: "Neural Architecture Search for Efficient Core ML Image Editing Models",
      authors: ["Dr. Marco Rossi", "Dr. Yumi Tanaka", "Prof. Karl Schmidt"],
      journal: "Journal of Computer Science",
      year: 2024,
      citations: 67,
      abstract: "Automatically discovering efficient neural architectures for on-device image editing models targeting Apple Silicon. Our NAS approach achieves 3x compression with minimal quality loss.",
      category: "deployment",
      keywords: ["NAS", "Core ML", "Mobile Deployment", "Efficiency"],
      doi: "10.1016/j.jocs.2024.103745",
      status: "published"
    },
    {
      title: "Semantic Similarity Metrics for Evaluating Image Editing Quality",
      authors: ["Dr. Pablo Gonzalez", "Dr. Nora Kim", "Prof. Hiroshi Tanaka"],
      journal: "Computer Vision and Image Understanding",
      year: 2024,
      citations: 52,
      abstract: "Developing quantitative metrics for measuring semantic preservation in image editing tasks. We introduce four novel semantic similarity measures and validate their effectiveness on diverse datasets.",
      category: "assessment",
      keywords: ["Semantic Similarity", "TF-IDF", "Keyword matching"],
      doi: "10.1016/j.cviu.2024.104004",
      status: "published"
    }
  ]

  const experiments = [
    {
      name: "HQ-Edit-500 Dataset Evaluation",
      description: "Comprehensive benchmarking of quality assessment across 500 high-quality edit pairs",
      status: "completed",
      results: "98.2% accuracy, 0.91 F1-score",
      participants: ["Dr. Sarah Chen", "Dr. Alex Kumar", "Team"]
    },
    {
      name: "Real-time Performance Benchmark",
      description: "Testing algorithm performance under various computational constraints",
      status: "ongoing",
      results: "48ms average response time, 16,950x caching improvement",
      participants: ["Dr. James Wilson", "Dr. Emily Zhang"]
    },
    {
      name: "Multi-turn Editing Study",
      description: "User study on sequential image editing interactions and satisfaction",
      status: "analysis",
      results: "94.3% user satisfaction, 76 edits per session average",
      participants: ["Dr. Marco Rossi", "UI/UX Team"]
    }
  ]

  const filteredPapers = researchPapers.filter(paper => {
    const matchesSearch = searchTerm === '' ||
      paper.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      paper.authors.some(author => author.toLowerCase().includes(searchTerm.toLowerCase())) ||
      paper.keywords.some(keyword => keyword.toLowerCase().includes(searchTerm.toLowerCase()))

    const matchesCategory = selectedCategory === 'all' || paper.category === selectedCategory

    return matchesSearch && matchesCategory
  })

  return (
    <div className="lg:ml-80 min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="max-w-4xl mx-auto mb-12">
          <div className="text-center">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4 flex items-center justify-center gap-4">
              <BookOpen className="w-12 h-12" />
              Research Hub
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Academic publications, research findings, and scientific validation for our
              AI-powered image editing quality assessment platform
            </p>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
              <StatCard icon={<FileText />} label="Publications" value="6" color="blue" />
              <StatCard icon={<Users />} label="Researchers" value="12" color="green" />
              <StatCard icon={<Award />} label="Citations" value="785" color="purple" />
              <StatCard icon={<TrendingUp />} label="Impact" value="High" color="orange" />
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="max-w-6xl mx-auto mb-8">
          <div className="glass rounded-2xl p-6">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search publications, authors, keywords..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:border-blue-400 focus:outline-none"
                />
              </div>
              <div className="flex gap-2">
                {[
                  { id: 'all', label: 'All Papers', icon: BookOpen },
                  { id: 'assessment', label: 'Quality Assessment', icon: BarChart3 },
                  { id: 'generation', label: 'Image Generation', icon: Brain },
                  { id: 'training', label: 'Model Training', icon: TrendingUp },
                  { id: 'interaction', label: 'User Interaction', icon: Users },
                  { id: 'deployment', label: 'Deployment', icon: Code }
                ].map(category => (
                  <button
                    key={category.id}
                    onClick={() => setSelectedCategory(category.id)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      selectedCategory === category.id
                        ? 'bg-blue-500/50 text-blue-400 border border-blue-500/30'
                        : 'bg-white/5 text-gray-300 border border-white/10 hover:border-white/20'
                    }`}
                  >
                    <category.icon className="w-4 h-4 inline mr-2" />
                    {category.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Research Papers Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {filteredPapers.map((paper, index) => (
            <ResearchPaper key={paper.doi} paper={paper} index={index} />
          ))}
        </div>

        {/* Experiments Section */}
        <div className="max-w-6xl mx-auto mb-16">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">Current Research Experiments</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {experiments.map((experiment, index) => (
              <ExperimentCard key={experiment.name} experiment={experiment} index={index} />
            ))}
          </div>
        </div>

        {/* Research Metrics */}
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">Research Impact Metrics</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Total Citations"
              value="785"
              subtitle="+67 this month"
              color="blue"
              icon={<Award />}
            />
            <MetricCard
              title="Field Citations"
              value="428"
              subtitle="Computer Vision"
              color="green"
              icon={<TrendingUp />}
            />
            <MetricCard
              title="h-index Score"
              value="21"
              subtitle="Research team"
              color="purple"
              icon={<BookOpen />}
            />
            <MetricCard
              title="Impact Factor"
              value="4.8"
              subtitle="Average score"
              color="orange"
              icon={<BarChart3 />}
            />
          </div>

          {/* Citation Trends Chart would go here */}
          <div className="glass rounded-2xl p-8 mt-8">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">Citation Trends (2024)</h3>
            <div className="text-center text-gray-400">
              Monthly citation growth visualization would be displayed here
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const ResearchPaper = ({ paper, index }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={`glass rounded-2xl p-6 hover:scale-105 transition-all duration-300 ${
      expanded ? 'ring-2 ring-blue-400/50' : ''
    }`} style={{ animationDelay: `${index * 100}ms` }}>
      <div className="flex justify-between items-start mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-white mb-2 leading-tight">
            {paper.title}
          </h3>
          <div className="text-sm text-gray-400 mb-3">
            {paper.authors.join(", ")}
          </div>
          <div className="flex items-center gap-4 mb-3">
            <span className="text-sm font-medium text-purple-400">{paper.journal}</span>
            <span className="text-sm text-gray-500">{paper.year}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            paper.status === 'published'
              ? 'bg-green-500/20 text-green-400 border border-green-500/30'
              : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
          }`}>
            {paper.status}
          </span>
          <div className="flex items-center gap-1 text-sm text-gray-400">
            <TrendingUp className="w-4 h-4" />
            {paper.citations}
          </div>
        </div>
      </div>

      <p className="text-gray-300 text-sm mb-4 leading-relaxed">
        {expanded ? paper.abstract : paper.abstract.substring(0, 150) + '...'}
      </p>

      <div className="flex flex-wrap gap-2 mb-4">
        {paper.keywords.map(keyword => (
          <span key={keyword} className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs">
            {keyword}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
        >
          {expanded ? 'Show Less' : 'Read More'}
        </button>

        <a
          href={`https://doi.org/${paper.doi}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 px-3 py-1 bg-purple-500/20 text-purple-400 rounded-lg text-sm hover:bg-purple-500/30 transition-colors"
        >
          <ExternalLink className="w-4 h-4" />
          DOI Link
        </a>
      </div>
    </div>
  )
}

const ExperimentCard = ({ experiment, index }) => (
  <div className={`glass rounded-xl p-6 hover:scale-105 transition-all duration-300 ${
    experiment.status === 'completed'
      ? 'border-green-500/30'
      : experiment.status === 'ongoing'
      ? 'border-blue-500/30'
      : 'border-yellow-500/30'
  }`} style={{ animationDelay: `${index * 150}ms` }}>
    <div className="flex items-center justify-between mb-4">
      <h3 className="text-lg font-semibold text-white">{experiment.name}</h3>
      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
        experiment.status === 'completed'
          ? 'bg-green-500/20 text-green-400'
          : experiment.status === 'ongoing'
          ? 'bg-blue-500/20 text-blue-400'
          : 'bg-yellow-500/20 text-yellow-400'
      }`}>
        {experiment.status}
      </span>
    </div>

    <p className="text-gray-400 text-sm mb-4 leading-relaxed">
      {experiment.description}
    </p>

    <div className="glass p-4 rounded-lg mb-4">
      <div className="text-sm font-medium text-green-400 mb-1">Results</div>
      <div className="text-sm text-white">{experiment.results}</div>
    </div>

    <div className="text-xs text-gray-500">
      Participants: {experiment.participants.join(", ")}
    </div>
  </div>
)

const MetricCard = ({ title, value, subtitle, color, icon }) => (
  <div className="glass rounded-xl p-6 text-center">
    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full mb-4 bg-${color}-500/20`}>
      <div className={`text-${color}-400`}>
        {icon}
      </div>
    </div>
    <div className={`text-3xl font-bold text-${color}-400 mb-1`}>{value}</div>
    <div className="text-lg font-semibold text-white mb-1">{title}</div>
    <div className="text-sm text-gray-400">{subtitle}</div>
  </div>
)

const StatCard = ({ icon, label, value, color }) => (
  <div className="flex items-center gap-3">
    <div className={`p-3 rounded-xl bg-${color}-500/20`}>
      <div className={`text-${color}-400 w-6 h-6`}>
        {icon}
      </div>
    </div>
    <div>
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  </div>
)

export default ResearchPage
