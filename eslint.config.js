import js from '@eslint/js'
import globals from 'globals'
import react from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'

export default [
  { ignores: ['node_modules/**', 'dist/**', 'coverage/**', '.venv/**'] },
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx,cjs,mjs}'],
    languageOptions: {
      ecmaVersion: 'latest',
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
  {
    files: ['src/**/*.{js,jsx}'],
    languageOptions: {
      globals: { ...globals.browser, __API_BASE_URL__: 'readonly' },
    },
    plugins: { react, 'react-hooks': reactHooks },
    settings: { react: { version: '18.2' } },
    rules: {
      ...react.configs.recommended.rules,
      ...react.configs['jsx-runtime'].rules,
      ...reactHooks.configs.recommended.rules,
      // Runtime prop validation is not used in this application.
      'react/prop-types': 'off',
    },
  },
  {
    files: ['*.{js,cjs,mjs}', '__mocks__/**/*.js'],
    languageOptions: { globals: globals.node },
  },
  {
    // Browser automation scripts also evaluate code in the page.
    files: ['scripts/**/*.{js,cjs,mjs}'],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
  {
    files: ['src/**/*.test.{js,jsx}', 'jest.setup.cjs'],
    languageOptions: { globals: { ...globals.jest, ...globals.browser, ...globals.node } },
  },
]
