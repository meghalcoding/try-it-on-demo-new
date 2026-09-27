import type { Product } from '../types/Product'
import type { LocalGLBUploadResult } from '../types/LocalGLBUpload'

const GLB_MAGIC = new Uint8Array([0x67, 0x6c, 0x54, 0x46])
const GLB_VERSION_2 = 2
const SAFE_NAME_PATTERN = /[<>:"/\\|?*\x00-\x1F]/g

export interface FileSystemDirectoryHandleLike {
  getFileHandle(name: string, options?: { create?: boolean }): Promise<FileSystemFileHandleLike>
}

interface FileSystemFileHandleLike {
  createWritable(): Promise<{ write(data: Blob): Promise<void>; close(): Promise<void> }>
}

interface WindowWithDirectoryPicker extends Window {
  showDirectoryPicker?: (options?: { mode?: 'read' | 'readwrite' }) => Promise<FileSystemDirectoryHandleLike>
}

export interface GLBValidationResult {
  readonly valid: boolean
  readonly message: string
}

export function sanitizeGLBName(name: string): string {
  const withoutExtension = name.trim().replace(/\.glb$/i, '')
  const sanitized = withoutExtension
    .replace(SAFE_NAME_PATTERN, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[. ]+$/g, '')
  return sanitized || 'untitled-glb'
}

export async function validateGLBFile(file: File): Promise<GLBValidationResult> {
  if (!file.name.toLowerCase().endsWith('.glb')) {
    return { valid: false, message: 'Only .glb files are allowed.' }
  }

  if (file.size < 12) {
    return { valid: false, message: 'The selected file is too small to be a valid GLB.' }
  }

  const header = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  const hasGLBMagic = GLB_MAGIC.every((byte, index) => header[index] === byte)
  const view = new DataView(header.buffer)
  const version = view.getUint32(4, true)
  const declaredLength = view.getUint32(8, true)

  if (!hasGLBMagic || version !== GLB_VERSION_2 || declaredLength > file.size || declaredLength < 12) {
    return { valid: false, message: 'The selected file is not a valid glTF 2.0 binary (.glb).' }
  }

  return { valid: true, message: '' }
}

export function canChooseLocalFolder(): boolean {
  return typeof (window as WindowWithDirectoryPicker).showDirectoryPicker === 'function'
}

export async function chooseLocalModelsFolder(): Promise<FileSystemDirectoryHandleLike> {
  const picker = (window as WindowWithDirectoryPicker).showDirectoryPicker
  if (!picker) {
    throw new Error('This browser does not support choosing a local folder. Use a current Chromium-based browser.')
  }

  return picker({ mode: 'readwrite' })
}

export async function saveGLBToLocalFolder(
  folder: FileSystemDirectoryHandleLike,
  file: File,
  displayName: string,
): Promise<string> {
  const baseName = sanitizeGLBName(displayName)
  const fileName = `${baseName}.glb`
  const fileHandle = await folder.getFileHandle(fileName, { create: true })
  const writable = await fileHandle.createWritable()
  await writable.write(file)
  await writable.close()
  return fileName
}

export function createUploadedProduct(
  file: File,
  displayName: string,
  sequence: number,
): { product: Product; result: LocalGLBUploadResult } {
  const safeName = sanitizeGLBName(displayName)
  const objectUrl = URL.createObjectURL(file)
  const productId = `uploaded-${safeName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'glb'}-${sequence}`

  const product: Product = Object.freeze({
    id: productId,
    name: safeName,
    price: 0,
    currency: 'USD',
    thumbnail: createGLBThumbnail(safeName),
    model: objectUrl,
    color: 'Uploaded GLB',
    size: 'Test asset',
    calibration: Object.freeze({
      scale: 1,
      x: 0,
      y: 0,
      z: 0,
      rotationX: 0,
      rotationY: 0,
      rotationZ: 0,
    }),
  })

  return {
    product,
    result: Object.freeze({
      fileName: `${safeName}.glb`,
      objectUrl,
      productId,
    }),
  }
}

function createGLBThumbnail(name: string): string {
  const escapedName = escapeXml(name.slice(0, 22))
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 220"><rect width="320" height="220" rx="24" fill="#202020"/><path d="M74 116c20-34 46-51 78-51s58 17 78 51c-20 9-39 14-58 14h-40c-19 0-38-5-58-14Z" fill="none" stroke="#f5f5f5" stroke-width="7"/><path d="M143 116h34M72 112l-25-13M248 112l25-13" stroke="#f5f5f5" stroke-width="7" stroke-linecap="round"/><text x="160" y="176" text-anchor="middle" fill="#f5f5f5" font-family="Arial,sans-serif" font-size="18" font-weight="700">${escapedName}</text><text x="160" y="198" text-anchor="middle" fill="#aaa" font-family="Arial,sans-serif" font-size="12">Uploaded GLB</text></svg>`
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`
}

function escapeXml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&apos;',
    '"': '&quot;',
  })[character] ?? character)
}
