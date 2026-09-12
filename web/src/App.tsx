import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Activity, Eye } from 'lucide-react'
import { UploadScreen } from '@/components/UploadScreen'
import { ResultView } from '@/components/ResultView'
import { SimulationDashboard } from '@/components/SimulationDashboard'
import type { ScreeningResponse } from '@/types'

type Tab = 'screen' | 'simulate'

function App() {
  const [tab, setTab] = useState<Tab>('screen')
  const [result, setResult] = useState<ScreeningResponse | null>(null)

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <Eye className="h-6 w-6 text-primary" />
            <span className="font-semibold tracking-tight">SugarEyes</span>
            <span className="text-sm text-muted-foreground">Explainable DR Screening</span>
          </div>
          <nav className="flex gap-1 rounded-lg bg-secondary p-1">
            <button
              onClick={() => setTab('screen')}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                tab === 'screen' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              Screen
            </button>
            <button
              onClick={() => setTab('simulate')}
              className={`flex items-center gap-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                tab === 'simulate' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Activity className="h-3.5 w-3.5" />
              Simulation
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <AnimatePresence mode="wait">
          {tab === 'screen' && (
            <motion.div
              key={result ? 'result' : 'upload'}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              {result ? (
                <ResultView result={result} onReset={() => setResult(null)} />
              ) : (
                <UploadScreen onResult={setResult} />
              )}
            </motion.div>
          )}

          {tab === 'simulate' && (
            <motion.div
              key="simulate"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              <SimulationDashboard />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App
