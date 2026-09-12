import { useCallback, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, Camera, Loader2, Upload } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { screenImage } from '@/lib/api'
import type { ScreeningResponse } from '@/types'

interface Props {
  onResult: (result: ScreeningResponse) => void
}

export function UploadScreen({ onResult }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [severity, setSeverity] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const pickFile = useCallback((f: File | undefined) => {
    if (!f) return
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setError(null)
  }, [])

  const handleSubmit = async () => {
    if (!file) return
    setSubmitting(true)
    setError(null)
    try {
      const result = await screenImage(file, severity)
      onResult(result)
    } catch {
      setError('Could not reach the screening API. Is it running at the configured address?')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-semibold tracking-tight">Upload a fundus photograph</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Colour retinal image, any common format. The pipeline handles cropping and preprocessing.
        </p>
      </div>

      <Card
        onDragOver={(e: React.DragEvent) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e: React.DragEvent) => {
          e.preventDefault()
          setDragOver(false)
          pickFile(e.dataTransfer.files[0])
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer border-2 border-dashed transition-colors ${
          dragOver ? 'border-primary bg-primary/5' : 'border-border'
        }`}
      >
        <CardContent className="flex flex-col items-center justify-center gap-3 py-12">
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => pickFile(e.target.files?.[0])}
          />
          {previewUrl ? (
            <motion.img
              key={previewUrl}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.2 }}
              src={previewUrl}
              alt="preview"
              className="h-48 w-48 rounded-full object-cover"
            />
          ) : (
            <>
              <Upload className="h-8 w-8 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Drag and drop, or click to choose a file</p>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Camera className="h-4 w-4 text-accent" />
            Simulate field capture
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            This project has no physical fundus camera. Dragging this slider deliberately degrades the
            uploaded image (blur, uneven illumination, haze, dust, exposure) to demonstrate how the
            quality gate and pipeline behave on realistic field conditions -- it is a stated stand-in,
            not a claim of real-world validation.
          </p>
          <div className="flex items-center gap-4">
            <Label className="w-24 shrink-0 font-mono text-xs text-muted-foreground">
              severity {severity.toFixed(2)}
            </Label>
            <Slider value={[severity]} onValueChange={(v) => setSeverity(v as number)} min={0} max={1} step={0.05} />
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Request failed</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Button size="lg" className="w-full" disabled={!file || submitting} onClick={handleSubmit}>
        {submitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" /> Screening...
          </>
        ) : (
          'Run screening'
        )}
      </Button>
    </div>
  )
}
