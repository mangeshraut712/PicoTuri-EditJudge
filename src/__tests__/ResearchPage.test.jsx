import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import ResearchPage from '../pages/ResearchPage'

test('filters publications by category and combines category with search', () => {
  render(<ResearchPage />)
  expect(screen.getAllByRole('link', { name: 'DOI Link' })).toHaveLength(6)
  fireEvent.click(screen.getByRole('button', { name: 'Quality Assessment' }))
  expect(screen.getAllByRole('link', { name: 'DOI Link' })).toHaveLength(2)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'semantic' } })
  expect(screen.getAllByRole('link', { name: 'DOI Link' })).toHaveLength(1)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'no matching publication' } })
  expect(screen.queryByRole('link', { name: 'DOI Link' })).not.toBeInTheDocument()
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: 'All Papers' }))
  expect(screen.getAllByRole('link', { name: 'DOI Link' })).toHaveLength(6)
})

test.each(['sarah CHEN', 'ResNet'])('searches authors and keywords without case sensitivity: %s', query => {
  render(<ResearchPage />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: query } })
  expect(screen.getAllByRole('link', { name: 'DOI Link' })).toHaveLength(1)
  expect(screen.getByRole('heading', { name: /Multi-Modal Image Editing/ })).toBeInTheDocument()
})

test('expands and collapses publication details', () => {
  render(<ResearchPage />)
  fireEvent.click(screen.getAllByRole('button', { name: 'Read More' })[0])
  fireEvent.click(screen.getByRole('button', { name: 'Show Less' }))
  expect(screen.queryByRole('button', { name: 'Show Less' })).not.toBeInTheDocument()
  expect(screen.getAllByRole('button', { name: 'Read More' })).toHaveLength(6)
})
