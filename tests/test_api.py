"""Regression tests for the demo API's public contract and local defaults."""

import os
import unittest
from unittest.mock import patch

from werkzeug.exceptions import TooManyRequests

from api import index


class ApiTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.app = index.create_app({'TESTING': True})
        self.client = self.app.test_client()

    def test_health_is_json_and_uses_milliseconds(self):
        with patch.object(index.time, 'time', return_value=1234.5):
            response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        self.assertEqual(response.json['status'], 'healthy')
        self.assertEqual(response.json['timestamp'], 1234500)

    def test_root_identifies_the_demo_service(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['data_source'], 'demo')

    def test_stats_do_not_claim_unmeasured_quality_or_coverage(self):
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['algorithms_count'], 7)
        self.assertEqual(response.json['data_source'], 'demo')
        for metric in ('code_quality', 'test_coverage', 'error_count',
                       'commit_count', 'last_updated'):
            with self.subTest(metric=metric):
                self.assertIsNone(response.json[metric])

    def test_all_algorithm_contracts_are_marked_as_demo(self):
        endpoints = {
            'quality-scorer': 'overall_score',
            'diffusion-model': 'inference_time_ms',
            'dpo-training': 'preference_accuracy',
            'multi-turn': 'edits_completed',
            'coreml': 'model_size_reduction',
            'baseline': 'validation_accuracy',
            'features': 'similarity_score',
        }
        for endpoint, metric in endpoints.items():
            with self.subTest(endpoint=endpoint):
                response = self.client.post('/api/test/' + endpoint, json={})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json['success'])
                self.assertEqual(response.json['data_source'], 'demo')
                self.assertIsInstance(response.json[metric], (int, float))

    def test_performance_metrics_are_identified_as_simulated(self):
        response = self.client.get('/api/performance/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['data_source'], 'demo')
        for metric in ('cpu_usage', 'memory_usage', 'requests_per_second',
                       'average_response_time', 'total_requests', 'error_rate'):
            with self.subTest(metric=metric):
                self.assertIsInstance(response.json[metric], (int, float))
                self.assertGreaterEqual(response.json[metric], 0)

    def test_performance_failure_does_not_expose_exception_details(self):
        with patch.object(index.random, 'uniform', side_effect=RuntimeError('private detail')):
            with self.assertLogs(index.logger, level='ERROR'):
                response = self.client.get('/api/performance/status')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {'error': 'Request failed'})
        self.assertNotIn('private detail', response.get_data(as_text=True))

    def test_unknown_routes_return_json(self):
        response = self.client.get('/api/missing')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json['status_code'], 404)
        self.assertIn('GET /api/health', response.json['available_endpoints'])

    def test_wrong_methods_return_json_and_preserve_allow_header(self):
        for method, endpoint, allowed in (
            ('get', '/api/test/baseline', 'POST'),
            ('post', '/api/health', 'GET'),
        ):
            with self.subTest(endpoint=endpoint):
                response = getattr(self.client, method)(endpoint)
                self.assertEqual(response.status_code, 405)
                self.assertEqual(response.json['status_code'], 405)
                self.assertEqual(response.json['error'], 'Method Not Allowed')
                self.assertIn(allowed, response.headers['Allow'])

    def test_unhandled_errors_return_generic_json(self):
        def fail():
            raise RuntimeError('private detail')

        self.app.config['PROPAGATE_EXCEPTIONS'] = False
        self.app.add_url_rule('/api/failure', view_func=fail)
        with self.assertLogs(self.app.logger, level='ERROR'):
            response = self.client.get('/api/failure')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {
            'error': 'Internal server error', 'status_code': 500
        })

    def test_http_errors_preserve_retry_headers(self):
        def rate_limited():
            raise TooManyRequests(retry_after=30)

        self.app.add_url_rule('/api/rate-limited', view_func=rate_limited)
        response = self.client.get('/api/rate-limited')
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['Retry-After'], '30')
        self.assertEqual(response.json, {
            'error': 'Too Many Requests', 'status_code': 429
        })

    def test_default_cors_allows_only_local_development_origins(self):
        for origin in ('http://localhost:3000', 'http://127.0.0.1:3000'):
            with self.subTest(origin=origin):
                response = self.client.get('/api/health', headers={'Origin': origin})
                self.assertEqual(response.headers['Access-Control-Allow-Origin'], origin)
                self.assertNotIn('Access-Control-Allow-Credentials', response.headers)

        for origin in ('https://attacker.example', 'http://localhost:3000.attacker.example',
                       'null', 'https://mangeshraut712.github.io'):
            with self.subTest(origin=origin):
                response = self.client.get('/api/health', headers={'Origin': origin})
                self.assertNotIn('Access-Control-Allow-Origin', response.headers)

    def test_cors_does_not_apply_to_root_or_requests_without_origin(self):
        response = self.client.get('/', headers={'Origin': 'http://localhost:3000'})
        self.assertNotIn('Access-Control-Allow-Origin', response.headers)
        response = self.client.get('/api/health')
        self.assertNotIn('Access-Control-Allow-Origin', response.headers)

    def test_preflight_allows_json_posts_and_rejects_unlisted_origins(self):
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        response = self.client.options('/api/test/baseline', headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn('POST', response.headers['Access-Control-Allow-Methods'])
        self.assertEqual(response.headers['Access-Control-Allow-Headers'], 'Content-Type')
        headers['Origin'] = 'https://attacker.example'
        response = self.client.options('/api/test/baseline', headers=headers)
        self.assertNotIn('Access-Control-Allow-Origin', response.headers)

    def test_preflight_does_not_allow_extra_methods_or_headers(self):
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'DELETE',
            'Access-Control-Request-Headers': 'Authorization'
        }
        response = self.client.options('/api/test/baseline', headers=headers)
        self.assertNotIn('Access-Control-Allow-Methods', response.headers)
        self.assertNotIn('Access-Control-Allow-Headers', response.headers)

        headers['Access-Control-Request-Method'] = 'POST'
        response = self.client.options('/api/test/baseline', headers=headers)
        self.assertNotIn('Access-Control-Allow-Headers', response.headers)

    def test_cors_headers_are_present_on_errors_for_allowed_origins(self):
        for endpoint, status in (('/api/missing', 404), ('/api/test/baseline', 405)):
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint, headers={'Origin': 'http://localhost:3000'})
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'http://localhost:3000')

    def test_cors_environment_can_restrict_access_to_a_deployment(self):
        with patch.dict(os.environ, {'CORS_ORIGINS': ' https://dashboard.example, https://other.example '}):
            client = index.create_app({'TESTING': True}).test_client()
        for origin in ('https://dashboard.example', 'https://other.example'):
            response = client.get('/api/health', headers={'Origin': origin})
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], origin)
        response = client.get('/api/health', headers={'Origin': 'http://localhost:3000'})
        self.assertNotIn('Access-Control-Allow-Origin', response.headers)

    def test_empty_cors_configuration_allows_same_origin_only(self):
        with patch.dict(os.environ, {'CORS_ORIGINS': ''}):
            client = index.create_app({'TESTING': True}).test_client()
        response = client.get('/api/health', headers={'Origin': 'http://localhost:3000'})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('Access-Control-Allow-Origin', response.headers)

    def test_cors_ignores_wildcards_credentials_and_origins_with_paths(self):
        configured = '*, null, https://user:secret@bad.example, https://ok.example/path, https://good.example/'
        with patch.dict(os.environ, {'CORS_ORIGINS': configured}):
            client = index.create_app({'TESTING': True}).test_client()
        response = client.get('/api/health', headers={'Origin': 'https://good.example'})
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'https://good.example')
        for origin in ('https://evil.example', 'https://ok.example', 'https://bad.example', 'null'):
            with self.subTest(origin=origin):
                response = client.get('/api/health', headers={'Origin': origin})
                self.assertNotIn('Access-Control-Allow-Origin', response.headers)

    def test_api_responses_are_not_cached_or_framed(self):
        for endpoint in ('/api/health', '/api/missing'):
            with self.subTest(endpoint=endpoint):
                response = self.client.get(endpoint)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
                self.assertEqual(response.headers['Referrer-Policy'], 'no-referrer')


class ServerConfigurationTests(unittest.TestCase):
    def test_server_binds_to_loopback_by_default_without_debugger(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(index.app, 'run') as run:
            self.assertEqual(index.main(), 0)
        run.assert_called_once_with(host='127.0.0.1', port=5001, debug=False, threaded=False)

    def test_server_honors_explicit_host_and_port(self):
        with patch.dict(os.environ, {'HOST': '0.0.0.0', 'PORT': '8123'}):
            with patch.object(index.app, 'run') as run:
                self.assertEqual(index.main(), 0)
        run.assert_called_once_with(host='0.0.0.0', port=8123, debug=False, threaded=False)

    def test_blank_host_stays_on_loopback(self):
        for host in ('', '   '):
            with self.subTest(host=host), patch.dict(os.environ, {'HOST': host}):
                with patch.object(index.app, 'run') as run:
                    self.assertEqual(index.main(), 0)
                run.assert_called_once_with(host='127.0.0.1', port=5001, debug=False, threaded=False)

    def test_host_and_port_whitespace_is_stripped(self):
        with patch.dict(os.environ, {'HOST': ' 0.0.0.0 ', 'PORT': ' 8123 '}):
            with patch.object(index.app, 'run') as run:
                self.assertEqual(index.main(), 0)
        run.assert_called_once_with(host='0.0.0.0', port=8123, debug=False, threaded=False)

    def test_invalid_ports_fail_before_starting_server(self):
        for port in ('not-a-port', '0', '-1', '65536'):
            with self.subTest(port=port), patch.dict(os.environ, {'PORT': port}):
                with patch.object(index.app, 'run') as run, self.assertLogs(index.logger, level='ERROR'):
                    self.assertEqual(index.main(), 1)
                run.assert_not_called()

    def test_startup_failure_returns_nonzero_status(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(index.app, 'run', side_effect=OSError('port unavailable')):
                with self.assertLogs(index.logger, level='ERROR'):
                    self.assertEqual(index.main(), 1)


if __name__ == '__main__':
    unittest.main()
