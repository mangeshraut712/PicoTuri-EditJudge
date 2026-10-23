"""
PicoTuri-EditJudge demo API backend
Lightweight Flask API for demonstration and testing
"""

from flask import Flask, jsonify, Blueprint, current_app
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import importlib
import logging
import os
import random
import time

# Optional import for type checking tools
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# API routes
@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - returns basic service status."""
    return jsonify({
        'status': 'healthy',
        'data_source': 'demo',
        'service': 'PicoTuri-EditJudge API',
        'version': '1.0.0',
        'timestamp': int(time.time() * 1000)
    }), 200

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Return endpoint counts without claiming unmeasured quality statistics."""
    return jsonify({
        'data_source': 'demo',
        'algorithms_count': 7,
        'code_quality': None,
        'test_coverage': None,
        'error_count': None,
        'commit_count': None,
        'last_updated': None
    }), 200

@api_bp.route('/performance/status', methods=['GET'])
def performance_status():
    """Return explicitly labeled, simulated performance metrics."""
    try:
        return jsonify({
            'data_source': 'demo',
            'timestamp': int(time.time() * 1000),
            'uptime': random.uniform(98.5, 99.9),
            'cpu_usage': random.uniform(20.0, 45.0),
            'memory_usage': random.uniform(35.0, 60.0),
            'requests_per_second': random.uniform(10.0, 35.0),
            'average_response_time': random.uniform(50.0, 120.0),
            'total_requests': random.randint(10000, 25000),
            'error_rate': random.uniform(0.005, 0.02),
            'algorithms_active': 7,
            'models_loaded': 3,
            'batch_queue_length': random.randint(0, 8),
            'disk_usage': random.uniform(30.0, 45.0),
            'network_in': random.uniform(2000, 8000),
            'network_out': random.uniform(8000, 20000),
            'services': {
                'api': 'healthy',
                'algorithms': 'all_active',
                'models': 'loaded',
                'cache': 'operational',
                'monitoring': 'active'
            }
        }), 200
    except Exception as exc:
        logger.error("Error in performance_status: %s", exc)
        return jsonify({'error': 'Request failed'}), 500

@api_bp.route('/test/quality-scorer', methods=['POST'])
def test_quality_scorer():
    """Return simulated quality scorer results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'overall_score': 0.91,
            'components': {
                'instruction_compliance': 0.92,
                'editing_realism': 0.89,
                'preservation_balance': 0.88,
                'technical_quality': 0.9
            },
            'grade': 'A',
            'recommendation': 'Excellent balance across fidelity and realism',
            'performance': {
                'inference_time_ms': 28.6,
                'throughput_images_per_min': 118,
                'latency_p99_ms': 36.4
            }
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_quality_scorer: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

@api_bp.route('/test/diffusion-model', methods=['POST'])
def test_diffusion_model():
    """Return simulated diffusion model results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'parameters': 10_900_000,
            'architecture': 'U-Net with cross-attention',
            'inference_time_ms': 83.2,
            'throughput_images_per_sec': 11.9,
            'quality_score': 4.6
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_diffusion_model: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

@api_bp.route('/test/dpo-training', methods=['POST'])
def test_dpo_training():
    """Return simulated DPO training results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'loss': 0.61,
            'preference_accuracy': 0.68,
            'training_steps': 12,
            'convergence_achieved': True
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_dpo_training: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

@api_bp.route('/test/multi-turn', methods=['POST'])
def test_multi_turn():
    """Return simulated multi-turn editor results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'instructions_processed': 3,
            'edits_completed': 3,
            'success_rate': 93.4,
            'processing_time_ms': 148.3
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_multi_turn: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

@api_bp.route('/test/coreml', methods=['POST'])
def test_coreml():
    """Return simulated Core ML optimizer results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'ios_files_generated': 3,
            'apple_silicon': True,
            'neural_engine_support': True,
            'quantization_applied': True,
            'coreml_version': '7.1',
            'target_ios_version': '17.0+',
            'model_size_reduction': 0.72,
            'compression_ratio': 3.6
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_coreml: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

@api_bp.route('/test/baseline', methods=['POST'])
def test_baseline():
    """Return simulated baseline model results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'classifier': 'LogisticRegression',
            'training_accuracy': 0.982,
            'validation_accuracy': 0.957,
            'roc_auc': 0.941,
            'f1_score': 0.924,
            'pipeline_steps': 3,
            'vocabulary_size': 1850
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_baseline: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

@api_bp.route('/test/features', methods=['POST'])
def test_features():
    """Return simulated feature extraction results."""
    try:
        response = {
            'data_source': 'demo',
            'success': True,
            'similarity_score': 0.87,
            'semantic_accuracy': 0.67,
            'tfidf_features': 1024,
            'vocabulary_size': 1850,
            'sparsity': 0.74,
            'between_group_similarity': 0.23,
            'within_group_similarity_brighten': 0.91,
            'within_group_similarity_darken': 0.89,
            'texts_processed': 6
        }
        return jsonify(response), 200
    except Exception as exc:
        logger.error("Error in test_features: %s", exc)
        return jsonify({'success': False, 'error': 'Request failed'}), 500

# Error handlers
def not_found(_: Exception) -> Any:
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'status_code': 404,
        'available_endpoints': [
            'GET /api/health',
            'GET /api/stats',
            'GET /api/performance/status',
            'POST /api/test/quality-scorer',
            'POST /api/test/diffusion-model',
            'POST /api/test/dpo-training',
            'POST /api/test/multi-turn',
            'POST /api/test/coreml',
            'POST /api/test/baseline',
            'POST /api/test/features'
        ]
    }), 404

def internal_error(_: Exception) -> Any:
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error', 'status_code': 500}), 500

def http_error(error: HTTPException) -> Any:
    """Keep HTTP status and headers while returning the API's JSON format."""
    response = error.get_response()
    response.data = current_app.json.dumps({
        'error': error.name,
        'status_code': error.code
    })
    response.content_type = 'application/json'
    return response

# Root endpoint
def root() -> Any:
    """Root endpoint for API information."""
    return jsonify({
        'data_source': 'demo',
        'name': 'PicoTuri-EditJudge API',
        'version': '1.0.0',
        'description': 'AI Quality Assessment Platform API',
        'endpoints': '/api/* routes available'
    }), 200

# Create app
def create_app(config=None):
    """Factory function to create Flask app."""
    app = Flask(__name__)
    app.config['CORS_ORIGINS'] = [
        origin.strip() for origin in os.environ.get(
            'CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000'
        ).split(',') if origin.strip()
    ]
    if config is not None:
        app.config.update(config)

    CORS(
        app,
        resources={r'/api/.*': {'origins': app.config['CORS_ORIGINS']}},
        methods=['GET', 'POST', 'OPTIONS'],
        allow_headers=['Content-Type'],
        supports_credentials=False,
        always_send=False
    )

    app.register_blueprint(api_bp, url_prefix='/api')
    app.add_url_rule('/', 'root', root)
    app.register_error_handler(404, not_found)
    app.register_error_handler(500, internal_error)
    app.register_error_handler(HTTPException, http_error)

    return app

app = create_app()

def main() -> int:
    """Run the development server on loopback unless explicitly configured."""
    logger.info("Starting PicoTuri-EditJudge API server...")
    logger.info("Available endpoints:\n"
                "  - GET  /api/health\n"
                "  - GET  /api/stats\n"
                "  - GET  /api/performance/status\n"
                "  - POST /api/test/* (7 test endpoints)")

    try:
        host = os.environ.get('HOST', '127.0.0.1')
        port = int(os.environ.get('PORT', '5001'))
        if not 1 <= port <= 65535:
            raise ValueError('PORT must be between 1 and 65535')
        logger.info("Server starting on http://%s:%s", host, port)
        app.run(host=host, port=port, debug=False, threaded=False)
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1

# Vercel WSGI handler (for deployment)
vercel_wsgi_module = None
try:
    vercel_wsgi_module = importlib.import_module('vercel_wsgi')
except ImportError:
    handler = None
else:
    handler = vercel_wsgi_module.handle_wsgi(app)

if __name__ == '__main__':
    raise SystemExit(main())
