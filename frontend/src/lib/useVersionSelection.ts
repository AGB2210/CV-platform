import { useState } from 'react'

/**
 * Multi-select state for a list of ids, shared by the dataset- and
 * model-version lists.
 *
 * Lives in its own module rather than beside the version components: a file
 * that exports both components and non-components breaks React Fast Refresh,
 * so editing the hook would force a full reload instead of a hot update.
 */
export function useVersionSelection() {
  const [selected, setSelected] = useState<Set<number>>(new Set())

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  /** Select-all is a toggle: clicking it when everything is already selected
   *  clears, which is what people expect from a header checkbox. */
  function toggleAll(ids: number[]) {
    setSelected((prev) => (prev.size === ids.length ? new Set() : new Set(ids)))
  }

  const clear = () => setSelected(new Set())

  return { selected, toggle, toggleAll, clear }
}
