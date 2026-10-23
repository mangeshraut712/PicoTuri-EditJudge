import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from '../App'

jest.mock('../pages/AlgorithmsPage', () => () => <h2>Algorithms page</h2>)
jest.mock('../pages/PerformanceDashboard', () => () => <h2>Performance page</h2>)
jest.mock('../pages/ResearchPage', () => () => <h2>Research page</h2>)

const renderApp = () => render(<MemoryRouter><App /></MemoryRouter>)
const openMenu = () => fireEvent.click(screen.getByRole('button', { name: 'Open navigation menu' }))
const expectClosed = () => {
  expect(screen.getByRole('button', { name: 'Open navigation menu' })).toHaveAttribute('aria-expanded', 'false')
  expect(screen.queryByRole('button', { name: 'Dismiss navigation menu' })).not.toBeInTheDocument()
}

test('opens and closes the menu repeatedly through the App state', () => {
  renderApp()
  expectClosed()
  for (let attempt = 0; attempt < 2; attempt += 1) {
    openMenu()
    const close = screen.getByRole('button', { name: 'Close navigation menu' })
    expect(close).toHaveAttribute('aria-expanded', 'true')
    expect(close).toHaveAttribute('aria-controls', 'primary-navigation')
    fireEvent.click(close)
    expectClosed()
  }
})

test('closes through the overlay and can be reopened', () => {
  renderApp()
  openMenu()
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss navigation menu' }))
  expectClosed()
  openMenu()
  expect(screen.getByRole('button', { name: 'Close navigation menu' })).toHaveAttribute('aria-expanded', 'true')
})

test('closes when navigating to a page or clicking the home logo', () => {
  renderApp()
  openMenu()
  fireEvent.click(screen.getByRole('link', { name: /Performance/ }))
  expect(screen.getByRole('heading', { name: 'Performance page' })).toBeInTheDocument()
  expectClosed()
  openMenu()
  fireEvent.click(screen.getByRole('link', { name: /PicoTuri/ }))
  expect(screen.getByRole('heading', { name: 'Algorithms page' })).toBeInTheDocument()
  expectClosed()
})
