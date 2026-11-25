/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi } from 'vitest'
import { focusableElements, trapTabKey } from '../utils/focusTrap'

const mountTrap = (innerHTML) => {
  document.body.innerHTML = `<div id="trap">${innerHTML}</div><button id="outside">outside</button>`
  return document.getElementById('trap')
}

const keyPress = (shiftKey = false) => ({ key: 'Tab', shiftKey, preventDefault: vi.fn() })

describe('focusableElements', () => {
  it('collects enabled focusable elements in document order, skipping disabled and tabindex=-1', () => {
    const container = mountTrap(`
      <a id="link" href="#a">a</a>
      <button disabled>disabled</button>
      <button id="btn">btn</button>
      <div tabindex="-1">skipped</div>
      <input id="field" type="text" />
      <div id="custom" tabindex="0">custom</div>
    `)
    expect(focusableElements(container).map(element => element.id)).toEqual(['link', 'btn', 'field', 'custom'])
  })
})

describe('trapTabKey', () => {
  it('wraps forward Tab from the last element to the first', () => {
    const container = mountTrap('<button id="first">first</button><button id="last">last</button>')
    const [first, last] = focusableElements(container)
    last.focus()
    const event = keyPress()
    trapTabKey(event, container)
    expect(event.preventDefault).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(first)
  })

  it('wraps backward Tab from the first element to the last', () => {
    const container = mountTrap('<button id="first">first</button><button id="last">last</button>')
    const [first, last] = focusableElements(container)
    first.focus()
    const event = keyPress(true)
    trapTabKey(event, container)
    expect(event.preventDefault).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(last)
  })

  it('leaves focus alone when it sits between the first and last element', () => {
    const container = mountTrap('<button id="a">a</button><button id="b">b</button><button id="c">c</button>')
    document.getElementById('b').focus()
    const event = keyPress()
    trapTabKey(event, container)
    expect(event.preventDefault).not.toHaveBeenCalled()
    expect(document.activeElement).toBe(document.getElementById('b'))
  })

  it('pulls focus back inside when it escaped the container', () => {
    const container = mountTrap('<button id="first">first</button><button id="last">last</button>')
    const [first, last] = focusableElements(container)
    document.getElementById('outside').focus()
    const forward = keyPress()
    trapTabKey(forward, container)
    expect(forward.preventDefault).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(first)
    document.getElementById('outside').focus()
    const backward = keyPress(true)
    trapTabKey(backward, container)
    expect(backward.preventDefault).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(last)
  })

  it('blocks Tab entirely when the container has no focusable elements', () => {
    const container = mountTrap('<p>nothing focusable</p>')
    const event = keyPress()
    trapTabKey(event, container)
    expect(event.preventDefault).toHaveBeenCalledTimes(1)
  })

  it('traps a single focusable element in both directions', () => {
    const container = mountTrap('<button id="only">only</button>')
    const only = document.getElementById('only')
    only.focus()
    const forward = keyPress()
    trapTabKey(forward, container)
    expect(forward.preventDefault).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(only)
    const backward = keyPress(true)
    trapTabKey(backward, container)
    expect(backward.preventDefault).toHaveBeenCalledTimes(1)
    expect(document.activeElement).toBe(only)
  })

  it('ignores non-Tab keys and missing containers', () => {
    const container = mountTrap('<button id="only">only</button>')
    const event = { key: 'Escape', shiftKey: false, preventDefault: vi.fn() }
    trapTabKey(event, container)
    expect(event.preventDefault).not.toHaveBeenCalled()
    trapTabKey(keyPress(), null)
  })
})
