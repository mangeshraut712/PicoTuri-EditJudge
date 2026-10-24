import { fireEvent, render, screen } from '@testing-library/react'
import ResearchPage from '../pages/ResearchPage'

const publications = () => screen.getAllByRole('article')

test('filters publications by category and combines category with search', () => {
  render(<ResearchPage />)
  expect(publications()).toHaveLength(6)
  fireEvent.click(screen.getByRole('button', { name: 'Quality Assessment' }))
  expect(screen.getByRole('button', { name: 'Quality Assessment' })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByRole('button', { name: 'All Papers' })).toHaveAttribute('aria-pressed', 'false')
  expect(publications()).toHaveLength(2)
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search publications' }), { target: { value: 'semantic' } })
  expect(publications()).toHaveLength(1)
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search publications' }), { target: { value: 'no matching publication' } })
  expect(screen.queryByRole('article')).not.toBeInTheDocument()
  expect(screen.getByText('No sample publications match this search.')).toBeInTheDocument()
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search publications' }), { target: { value: '' } })
  fireEvent.click(screen.getByRole('button', { name: 'All Papers' }))
  expect(publications()).toHaveLength(6)
})

test.each(['sarah CHEN', 'ResNet'])('searches authors and keywords without case sensitivity: %s', query => {
  render(<ResearchPage />)
  fireEvent.change(screen.getByRole('searchbox', { name: 'Search publications' }), { target: { value: query } })
  expect(publications()).toHaveLength(1)
  expect(screen.getByRole('heading', { name: /Multi-Modal Image Editing/ })).toBeInTheDocument()
})

test('labels the bibliography as sample data and does not link fabricated DOIs', () => {
  render(<ResearchPage />)
  expect(screen.getByRole('note')).toHaveTextContent('Sample bibliography')
  expect(screen.getByText('DOI 10.1109/CVPR57869.2024.01234')).toBeInTheDocument()
  expect(screen.queryByRole('link')).not.toBeInTheDocument()
  expect(screen.getByText('No citation trend data is available for this sample bibliography.')).toBeInTheDocument()
})

test('expands and collapses publication details', () => {
  render(<ResearchPage />)
  fireEvent.click(screen.getAllByRole('button', { name: 'Read More' })[0])
  fireEvent.click(screen.getByRole('button', { name: 'Show Less' }))
  expect(screen.queryByRole('button', { name: 'Show Less' })).not.toBeInTheDocument()
  expect(screen.getAllByRole('button', { name: 'Read More' })).toHaveLength(6)
})
