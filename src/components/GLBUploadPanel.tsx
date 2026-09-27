import { useRef, useState } from 'react'
import { canChooseLocalFolder, saveGLBToLocalFolder, validateGLBFile, type FileSystemDirectoryHandleLike } from '../products/LocalGLBUploadService'

export interface GLBUploadPanelProps {
  readonly folderSelected: boolean
  readonly onChooseFolder: () => Promise<void>
  readonly onUpload: (file: File, displayName: string) => Promise<void>
}

export function GLBUploadPanel({ folderSelected, onChooseFolder, onUpload }: GLBUploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [displayName, setDisplayName] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [folderReady, setFolderReady] = useState(folderSelected)

  const chooseFolder = async () => {
    setError('')
    setStatus('')
    try {
      await onChooseFolder()
      setFolderReady(true)
      setStatus('Models folder selected. New GLBs will be saved there.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The local models folder could not be selected.')
    }
  }

  const chooseFile = async (file: File | undefined) => {
    setError('')
    setStatus('')
    setSelectedFile(null)

    if (!file) return

    try {
      const validation = await validateGLBFile(file)
      if (!validation.valid) {
        setError(validation.message)
        if (fileInputRef.current) fileInputRef.current.value = ''
        return
      }

      setSelectedFile(file)
      setDisplayName(file.name.replace(/\.glb$/i, ''))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The GLB file could not be validated.')
    }
  }

  const upload = async () => {
    if (!selectedFile || !displayName.trim()) {
      setError('Choose a GLB file and enter a name before uploading.')
      return
    }

    if (!folderReady) {
      setError('Choose the local models folder before saving the GLB.')
      return
    }

    setSaving(true)
    setError('')
    setStatus('Saving GLB…')
    try {
      await onUpload(selectedFile, displayName.trim())
      setStatus(`${displayName.trim()}.glb saved and added to the carousel.`)
      setSelectedFile(null)
      setDisplayName('')
      if (fileInputRef.current) fileInputRef.current.value = ''
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The GLB could not be saved.')
      setStatus('')
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="glb-upload-panel" aria-label="Test a new GLB model">
      <div className="glb-upload-panel__header">
        <div>
          <p className="eyebrow">Developer testing</p>
          <h2>Upload GLB</h2>
        </div>
        <span className={`glb-upload-panel__status${folderReady ? ' glb-upload-panel__status--ready' : ''}`}>
          {folderReady ? 'Folder ready' : 'Folder needed'}
        </span>
      </div>

      <p className="glb-upload-panel__description">
        Select your local public/assets/models folder, then test a new binary glTF model. Only valid glTF 2.0 .glb files are accepted.
      </p>

      <div className="glb-upload-panel__actions">
        {canChooseLocalFolder() && (
          <button type="button" className="calibration-button calibration-button--secondary" onClick={chooseFolder} disabled={saving}>
            {folderReady ? 'Change Models Folder' : 'Choose Models Folder'}
          </button>
        )}
        <label className="glb-upload-panel__file-button">
          <span>Choose GLB</span>
          <input
            ref={fileInputRef}
            type="file"
            accept=".glb,model/gltf-binary"
            onChange={(event) => void chooseFile(event.target.files?.[0])}
            disabled={saving}
          />
        </label>
      </div>

      <label className="glb-upload-panel__field">
        <span>File name</span>
        <input
          type="text"
          value={displayName}
          onChange={(event) => setDisplayName(event.target.value)}
          placeholder="e.g. test-frame-001"
          disabled={saving}
          maxLength={120}
        />
      </label>

      {selectedFile && <p className="glb-upload-panel__file">Selected: {selectedFile.name}</p>}

      <button type="button" className="calibration-button glb-upload-panel__save" onClick={() => void upload()} disabled={saving || !selectedFile || !displayName.trim() || !folderReady}>
        {saving ? 'Saving…' : 'Save & Test GLB'}
      </button>

      {status && <p className="glb-upload-panel__message" role="status">{status}</p>}
      {error && <p className="glb-upload-panel__error" role="alert">{error}</p>}
    </section>
  )
}

export function saveSelectedGLBToFolder(
  folder: FileSystemDirectoryHandleLike,
  file: File,
  displayName: string,
) {
  return saveGLBToLocalFolder(folder, file, displayName)
}
