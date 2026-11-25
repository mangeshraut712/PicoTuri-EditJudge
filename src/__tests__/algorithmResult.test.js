import { describe, expect, it } from 'vitest'
import { isValidAlgorithmResult } from '../components/algorithms/algorithmResult'
import { algorithmResults } from './algorithmFixtures'

describe('isValidAlgorithmResult', () => {
  it('accepts every canonical algorithm payload', () => {
    Object.entries(algorithmResults).forEach(([name, result]) => {
      expect(isValidAlgorithmResult(name, result)).toBe(true)
    })
  })

  it('rejects unknown names, including inherited object keys', () => {
    expect(isValidAlgorithmResult('Unknown Algorithm', algorithmResults['DPO Training'])).toBe(false)
    expect(isValidAlgorithmResult('constructor', { success: true })).toBe(false)
    expect(isValidAlgorithmResult('toString', { success: true })).toBe(false)
  })

  it('requires success to be exactly the boolean true', () => {
    const dpo = algorithmResults['DPO Training']
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, success: 'true' })).toBe(false)
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, success: 1 })).toBe(false)
  })

  it('enforces inclusive bounds on fractions, similarities, scores and rates', () => {
    const dpo = algorithmResults['DPO Training']
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, preference_accuracy: 0 })).toBe(true)
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, preference_accuracy: 1 })).toBe(true)
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, preference_accuracy: 1.01 })).toBe(false)
    const features = algorithmResults['Feature Extraction']
    expect(isValidAlgorithmResult('Feature Extraction', { ...features, similarity_score: -1 })).toBe(true)
    expect(isValidAlgorithmResult('Feature Extraction', { ...features, similarity_score: 1.2 })).toBe(false)
    const diffusion = algorithmResults['Diffusion Model']
    expect(isValidAlgorithmResult('Diffusion Model', { ...diffusion, quality_score: 5 })).toBe(true)
    expect(isValidAlgorithmResult('Diffusion Model', { ...diffusion, quality_score: 5.1 })).toBe(false)
    const multiTurn = algorithmResults['Multi-Turn Editor']
    expect(isValidAlgorithmResult('Multi-Turn Editor', { ...multiTurn, success_rate: 100 })).toBe(true)
    expect(isValidAlgorithmResult('Multi-Turn Editor', { ...multiTurn, success_rate: 100.1 })).toBe(false)
  })

  it('requires counts to be non-negative integers and timings to be non-negative', () => {
    const diffusion = algorithmResults['Diffusion Model']
    expect(isValidAlgorithmResult('Diffusion Model', { ...diffusion, parameters: 10900000.5 })).toBe(false)
    expect(isValidAlgorithmResult('Diffusion Model', { ...diffusion, inference_time_ms: -1 })).toBe(false)
    const multiTurn = algorithmResults['Multi-Turn Editor']
    expect(isValidAlgorithmResult('Multi-Turn Editor', { ...multiTurn, edits_completed: 2.5 })).toBe(false)
  })

  it('requires a non-empty, in-range components map for the Quality Scorer', () => {
    const scorer = algorithmResults['Quality Scorer']
    expect(isValidAlgorithmResult('Quality Scorer', { ...scorer, components: {} })).toBe(false)
    expect(isValidAlgorithmResult('Quality Scorer', { ...scorer, components: { instruction_compliance: 1.5 } })).toBe(false)
    const withoutComponents = { ...scorer }
    delete withoutComponents.components
    expect(isValidAlgorithmResult('Quality Scorer', withoutComponents)).toBe(false)
  })

  it('validates training_history shape, lengths and value ranges', () => {
    const dpo = algorithmResults['DPO Training']
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, training_history: { loss: [0.9, 0.61], accuracy: [55, 68] } })).toBe(true)
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, training_history: { loss: [0.9], accuracy: [55, 68] } })).toBe(false)
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, training_history: { loss: [], accuracy: [] } })).toBe(false)
    expect(isValidAlgorithmResult('DPO Training', { ...dpo, training_history: { loss: [0.9], accuracy: [101] } })).toBe(false)
  })
})
