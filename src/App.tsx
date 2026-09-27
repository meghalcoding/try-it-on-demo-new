import { useEffect, useRef, useState } from 'react'
import './App.css'
import { CameraError, CameraManager } from './ar/CameraManager'
import { FaceTracker } from './ar/FaceTracker'
import { FaceMeshOverlay } from './ar/FaceMeshOverlay'
import { FaceTrackingState, type TrackingState } from './ar/FaceTrackingState'
import { VideoFrameScheduler } from './ar/VideoFrameScheduler'
import { FaceDebugOverlay } from './ar/FaceDebugOverlay'
import { ARRenderer } from './ar/ARRenderer'
import { ForegroundSegmenter } from './ar/occlusion/ForegroundSegmenter'
import { LightingSampler } from './ar/occlusion/LightEstimator'
import { GLBLoader, disposeLoadedGLB, type LoadedGLB } from './ar/GLBLoader'
import { CalibrationPanel, calibrationToSliderValues, identityCalibration } from './components/CalibrationPanel'
import { DEFAULT_SMOOTHING_SETTINGS, SmoothingPanel, smoothingToSliderValues } from './components/SmoothingPanel'
import { FaceOcclusionPanel, faceOcclusionToSliderValues, type OcclusionStatusView } from './components/FaceOcclusionPanel'
import type { PoseSmoothingSettings } from './ar/PoseSmoother'
import type { Calibration } from './types/Calibration'
import { DEFAULT_FACE_OCCLUSION_SETTINGS, type FaceOcclusionSettings } from './types/FaceOcclusion'
import { logCalibrationDiagnostics } from './utils/CalibrationDiagnostics'
import { mediaPipeTransformationMatrixToFacePose } from './ar/coordinateTransform'
import { isMediaPipeFaceTransformationMatrix } from './ar/FacePose'
import { ProductCarousel } from './components/ProductCarousel'
import { GLBUploadPanel, saveSelectedGLBToFolder } from './components/GLBUploadPanel'
import { createUploadedProduct, chooseLocalModelsFolder } from './products/LocalGLBUploadService'
import { ProductService } from './products/ProductService'
import { GLBAssetAnalyzer } from './ar/glb/GLBAssetAnalyzer'
import { AutoCalibrationEngine } from './ar/glb/AutoCalibrationEngine'
import { GLBDiagnosticPanel } from './components/GLBDiagnosticPanel'
import type { AutoCalibrationResult, GLBAssetAnalysis } from './ar/glb/types'

type CameraUiState = 'landing' | 'requesting' | 'active' | 'denied' | 'unavailable'

const productService = new ProductService()
const glbAnalyzer = new GLBAssetAnalyzer()
const autoCalibrationEngine = new AutoCalibrationEngine()

function TryOnView() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const cameraManagerRef = useRef<CameraManager | null>(null)
  const faceTrackerRef = useRef<FaceTracker | null>(null)
  const trackingStateRef = useRef(new FaceTrackingState())
  const meshOverlayRef = useRef<FaceMeshOverlay | null>(null)
  const frameSchedulerRef = useRef<VideoFrameScheduler | null>(null)
  const faceDebugOverlayRef = useRef<FaceDebugOverlay | null>(null)
  const arRendererRef = useRef<ARRenderer | null>(null)
  const glbLoaderRef = useRef<GLBLoader | null>(null)
  const segmenterRef = useRef<ForegroundSegmenter | null>(null)
  const lightingSamplerRef = useRef(new LightingSampler())
  const faceOcclusionSettingsRef = useRef<FaceOcclusionSettings>({ ...DEFAULT_FACE_OCCLUSION_SETTINGS })
  const arCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const [cameraState, setCameraState] = useState<CameraUiState>('landing')
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
  const [cameraMessage, setCameraMessage] = useState('')
  const [trackingState, setTrackingState] = useState<TrackingState>('searching')
  const [showFaceMesh, setShowFaceMesh] = useState(false)
  const [trackingError, setTrackingError] = useState('')
  const [arError, setArError] = useState('')
  const [calibration, setCalibration] = useState<Calibration>(() => identityCalibration())
  const [autoCalibrationResult, setAutoCalibrationResult] = useState<AutoCalibrationResult | null>(null)
  const [smoothingSettings, setSmoothingSettings] = useState<PoseSmoothingSettings>(() => ({ ...DEFAULT_SMOOTHING_SETTINGS }))
  const [faceOcclusionSettings, setFaceOcclusionSettings] = useState<FaceOcclusionSettings>(() => ({ ...DEFAULT_FACE_OCCLUSION_SETTINGS }))
  const [occlusionStatus, setOcclusionStatus] = useState<OcclusionStatusView | null>(null)
  const [, setProductCatalogVersion] = useState(0)
  const products = productService.listProducts()
  const [selectedProductId, setSelectedProductId] = useState(() => products[0]?.id ?? '')
  const [loadingProductId, setLoadingProductId] = useState<string | null>(null)
  const lastUiTrackingStateRef = useRef<TrackingState>('searching')
  const latestFacePoseRef = useRef<ReturnType<typeof mediaPipeTransformationMatrixToFacePose> | null>(null)
  const activeLoadedModelRef = useRef<LoadedGLB | null>(null)
  const modelsFolderRef = useRef<Awaited<ReturnType<typeof chooseLocalModelsFolder>> | null>(null)
  const uploadedObjectUrlsRef = useRef<string[]>([])
  const uploadSequenceRef = useRef(0)

  useEffect(() => {
    if (cameraState !== 'active' || !arCanvasRef.current) {
      return
    }

    const renderer = new ARRenderer(arCanvasRef.current, {
      onContextError: setArError,
      onRuntimeError: setArError,
    })
    const glbLoader = new GLBLoader()
    arRendererRef.current = renderer
    glbLoaderRef.current = glbLoader
    renderer.setGlassesCalibration(calibration)
    renderer.setPoseSmoothingSettings(smoothingSettings)
    renderer.setFaceOcclusionSettings(faceOcclusionSettings)
    renderer.setFacePose(latestFacePoseRef.current)
    renderer.start()

    return () => {
      renderer.setFacePose(null)
      renderer.clearModel()
      renderer.dispose()
      if (activeLoadedModelRef.current) {
        disposeLoadedGLB(activeLoadedModelRef.current)
        activeLoadedModelRef.current = null
      }
      glbLoader.dispose()
      glbLoaderRef.current = null
      arRendererRef.current = null
    }
  }, [cameraState])

  useEffect(() => {
    if (cameraState !== 'active' || !selectedProductId) {
      return
    }

    const renderer = arRendererRef.current
    const glbLoader = glbLoaderRef.current
    if (!renderer || !glbLoader) {
      return
    }

    let cancelled = false
    let product: ReturnType<ProductService['getProductById']>

    try {
      product = productService.getProductById(selectedProductId)
    } catch (error) {
      setArError(error instanceof Error ? error.message : 'The selected eyewear product could not be resolved.')
      setLoadingProductId(null)
      return
    }

    setArError('')
    setLoadingProductId(product.id)

    void glbLoader.load(product.model)
      .then((loadedModel) => {
        if (cancelled || arRendererRef.current !== renderer || glbLoaderRef.current !== glbLoader) {
          disposeLoadedGLB(loadedModel)
          return
        }

        const previousModel = activeLoadedModelRef.current

        // Perform single-pass GLB geometry analysis & auto calibration
        try {
          const analysis = glbAnalyzer.analyzeAsset(loadedModel.scene)
          const result = autoCalibrationEngine.computeAutoCalibration(analysis, null)
          setAutoCalibrationResult(result)

          const activeCalibration = result.isAutoApplied ? result.finalCalibration : product.calibration
          renderer.setGlassesCalibration(activeCalibration)
          renderer.setModel(result.normalization.runtimeRoot)
          activeLoadedModelRef.current = loadedModel
          setCalibration(activeCalibration)
        } catch (analysisErr) {
          // Fallback if analysis fails on unusual mesh
          renderer.setGlassesCalibration(product.calibration)
          renderer.setModel(loadedModel.scene)
          activeLoadedModelRef.current = loadedModel
          setCalibration(product.calibration)
          setAutoCalibrationResult(null)
        }

        setLoadingProductId(null)

        if (previousModel && previousModel !== loadedModel) {
          disposeLoadedGLB(previousModel)
        }
      })
      .catch((error) => {
        if (cancelled || arRendererRef.current !== renderer || glbLoaderRef.current !== glbLoader) {
          return
        }

        setLoadingProductId(null)
        setArError(
          error instanceof Error
            ? `Eyewear model could not be loaded: ${error.message}`
            : 'Eyewear model could not be loaded.',
        )
      })

    return () => {
      cancelled = true
    }
  }, [cameraState, selectedProductId])

  useEffect(() => {
    faceOcclusionSettingsRef.current = faceOcclusionSettings
  }, [faceOcclusionSettings])

  useEffect(() => {
    if (cameraState !== 'active') {
      setOcclusionStatus(null)
      return
    }

    // Only publish when something the panel shows actually changed: publishing
    // re-renders this component, which re-runs the inline canvas ref callbacks.
    let lastSignature = ''
    const id = window.setInterval(() => {
      const renderer = arRendererRef.current
      if (!renderer) return
      const segmenter = segmenterRef.current
      const occlusion = renderer.getOcclusionStatus()
      const view: OcclusionStatusView = {
        occlusion,
        segmenter: segmenter?.getStatus() ?? 'idle',
        segmenterMessage: segmenter?.getStatusMessage() ?? '',
        segmenterMs: segmenter?.getAverageDurationMs() ?? 0,
      }
      const signature = [
        occlusion.occluder.mode,
        occlusion.occluder.hasSurface,
        occlusion.occluder.confidence.toFixed(2),
        occlusion.fit.verdict,
        occlusion.fit.embeddedFraction.toFixed(2),
        occlusion.fit.suggestedForwardCm.toFixed(1),
        occlusion.appliedClearanceCm.toFixed(1),
        occlusion.foregroundMaskActive,
        occlusion.cameraFovDeg.toFixed(1),
        view.segmenter,
        view.segmenterMessage,
        Math.round(view.segmenterMs),
      ].join('|')
      if (signature !== lastSignature) {
        lastSignature = signature
        setOcclusionStatus(view)
      }
    }, 500)

    return () => window.clearInterval(id)
  }, [cameraState])

  useEffect(() => {
    const initialCalibration = identityCalibration()
    logCalibrationDiagnostics(initialCalibration, {
      source: 'init',
      sliderValues: {
        ...calibrationToSliderValues(initialCalibration),
        ...smoothingToSliderValues(DEFAULT_SMOOTHING_SETTINGS),
        ...faceOcclusionToSliderValues(DEFAULT_FACE_OCCLUSION_SETTINGS),
      },
      smoothing: DEFAULT_SMOOTHING_SETTINGS,
      occlusion: DEFAULT_FACE_OCCLUSION_SETTINGS,
    })
  }, [])

  useEffect(() => {
    const manager = new CameraManager()
    cameraManagerRef.current = manager

    const tracker = new FaceTracker()
    const meshOverlay = new FaceMeshOverlay()
    const debugOverlay = new FaceDebugOverlay()
    faceTrackerRef.current = tracker
    segmenterRef.current = new ForegroundSegmenter()
    meshOverlayRef.current = meshOverlay
    faceDebugOverlayRef.current = debugOverlay

    if (videoRef.current) {
      manager.attachVideoElement(videoRef.current)
    }

    return () => {
      arRendererRef.current?.dispose()
      arRendererRef.current = null
      glbLoaderRef.current?.dispose()
      glbLoaderRef.current = null
      frameSchedulerRef.current?.dispose()
      frameSchedulerRef.current = null
      tracker.dispose()
      segmenterRef.current?.dispose()
      segmenterRef.current = null
      meshOverlay.dispose()
      debugOverlay.dispose()
      manager.dispose()
      cameraManagerRef.current = null
      faceTrackerRef.current = null
      meshOverlayRef.current = null
      faceDebugOverlayRef.current = null
      for (const objectUrl of uploadedObjectUrlsRef.current) {
        URL.revokeObjectURL(objectUrl)
      }
      uploadedObjectUrlsRef.current = []
    }
  }, [])

  const stopTrackingPreview = () => {
    frameSchedulerRef.current?.stop()
    frameSchedulerRef.current = null
  }

  const startInference = async () => {
    const video = videoRef.current
    const tracker = faceTrackerRef.current
    if (!video || !tracker) return

    setTrackingError('')
    try {
      await tracker.initialize()
    } catch (error) {
      setTrackingError(error instanceof Error ? error.message : 'Face tracking could not be initialized.')
      return
    }

    stopTrackingPreview()
    trackingStateRef.current.reset()
    lastUiTrackingStateRef.current = 'searching'
    setTrackingState('searching')

    const scheduler = new VideoFrameScheduler(video, tracker, {
      onFrame: (result) => {
        arRendererRef.current?.setVideoSourceDimensions(video.videoWidth, video.videoHeight)
        const selection = trackingStateRef.current.update(result.result)

        if (selection.state !== lastUiTrackingStateRef.current) {
          lastUiTrackingStateRef.current = selection.state
          setTrackingState(selection.state)
        }

        const primaryFaceIndex = selection.primaryFaceIndex
        const transformationMatrix = primaryFaceIndex === null
          ? undefined
          : result.result.facialTransformationMatrixes[primaryFaceIndex]

        if (
          selection.state === 'detected' &&
          transformationMatrix &&
          isMediaPipeFaceTransformationMatrix(transformationMatrix)
        ) {
          try {
            const facePose = mediaPipeTransformationMatrixToFacePose(
              transformationMatrix,
              selection.state,
              result.timestampMs,
            )
            latestFacePoseRef.current = facePose
            arRendererRef.current?.setFacePose(facePose)
          } catch (error) {
            latestFacePoseRef.current = null
            arRendererRef.current?.setFacePose(null)
            arRendererRef.current?.setFaceLandmarks(null)
            setTrackingError(
              error instanceof Error
                ? error.message
                : 'Face pose conversion failed.',
            )
          }
        } else {
          latestFacePoseRef.current = null
          arRendererRef.current?.setFacePose(null)
          arRendererRef.current?.setFaceLandmarks(null)
        }

        const selectedLandmarks = selection.primaryFaceIndex === null
          ? null
          : result.result.faceLandmarks[selection.primaryFaceIndex] ?? null
        arRendererRef.current?.setFaceLandmarks(
          selection.state === 'detected' ? selectedLandmarks : null,
        )

        // Optional layers. Both are lazy and non-blocking: face tracking and the
        // depth occluder never wait for them.
        const occlusionSettings = faceOcclusionSettingsRef.current
        if (selection.state === 'detected' && selectedLandmarks && occlusionSettings.enabled) {
          const now = performance.now()

          if (occlusionSettings.foregroundMaskEnabled) {
            const segmenter = segmenterRef.current
            if (segmenter) {
              if (segmenter.getStatus() === 'idle') void segmenter.initialize()
              const mask = segmenter.process(video, selectedLandmarks, now)
              if (mask) arRendererRef.current?.setForegroundMask(mask)
            }
          }

          if (occlusionSettings.lightingMatchEnabled) {
            const sample = lightingSamplerRef.current.sample(video, selectedLandmarks, now)
            if (sample) arRendererRef.current?.setLightingSample(sample)
          }
        } else {
          arRendererRef.current?.setForegroundMask(null)
        }

        meshOverlayRef.current?.render(
          result,
          selection.primaryFaceIndex,
          video,
          selection.bounds,
        )
        faceDebugOverlayRef.current?.render(
          result,
          selection.primaryFaceIndex,
          selection.bounds,
          latestFacePoseRef.current,
        )
      },
      onError: (error) => {
        setTrackingError(error instanceof Error ? error.message : 'Face tracking failed.')
      },
    })

    frameSchedulerRef.current = scheduler
    scheduler.start()
  }

  const toggleFaceMesh = () => {
    const nextValue = !showFaceMesh
    setShowFaceMesh(nextValue)

    if (!nextValue) {
      meshOverlayRef.current?.clear()
    }
  }

  const startCamera = async () => {
    const manager = cameraManagerRef.current

    if (!manager) {
      setCameraState('unavailable')
      setCameraMessage('The camera service is not available. Please reload the page and try again.')
      return
    }

    setCameraState('requesting')
    setCameraMessage('Requesting camera access…')

    try {
      await manager.start()
      setCameraState('active')
      setCameraMessage('Camera access is active.')
      await startInference()
    } catch (error) {
      const cameraError = error instanceof CameraError ? error : null

      if (cameraError?.code === 'permission-denied') {
        setCameraState('denied')
        setCameraMessage('Camera permission is blocked. Enable camera access in your browser settings, then retry.')
        return
      }

      setCameraState('unavailable')
      setCameraMessage(
        cameraError?.message ?? 'The camera could not be started. Check your browser and camera, then try again.',
      )
    }
  }

  const updateCalibration = (
    nextCalibration: Calibration,
    source: 'change' | 'reset' = 'change',
  ) => {
    setCalibration(nextCalibration)
    arRendererRef.current?.setGlassesCalibration(nextCalibration)

    logCalibrationDiagnostics(nextCalibration, {
      source,
      sliderValues: {
        ...calibrationToSliderValues(nextCalibration),
        ...smoothingToSliderValues(smoothingSettings),
        ...faceOcclusionToSliderValues(faceOcclusionSettings),
      },
      smoothing: smoothingSettings,
      occlusion: faceOcclusionSettings,
    })
  }

  const resetCalibration = () => {
    const nextCalibration = identityCalibration()
    updateCalibration(nextCalibration, 'reset')
  }

  const updateSmoothingSettings = (nextSettings: PoseSmoothingSettings, source: 'change' | 'reset' = 'change') => {
    setSmoothingSettings(nextSettings)
    arRendererRef.current?.setPoseSmoothingSettings(nextSettings)
    logCalibrationDiagnostics(calibration, {
      source: source === 'reset' ? 'reset' : 'change',
      sliderValues: {
        ...calibrationToSliderValues(calibration),
        ...smoothingToSliderValues(nextSettings),
        ...faceOcclusionToSliderValues(faceOcclusionSettings),
      },
      smoothing: nextSettings,
      occlusion: faceOcclusionSettings,
    })
  }

  const resetSmoothing = () => {
    updateSmoothingSettings({ ...DEFAULT_SMOOTHING_SETTINGS }, 'reset')
  }

  const updateFaceOcclusion = (nextSettings: FaceOcclusionSettings, source: 'change' | 'reset' = 'change') => {
    setFaceOcclusionSettings(nextSettings)
    arRendererRef.current?.setFaceOcclusionSettings(nextSettings)
    logCalibrationDiagnostics(calibration, {
      source,
      sliderValues: {
        ...calibrationToSliderValues(calibration),
        ...smoothingToSliderValues(smoothingSettings),
        ...faceOcclusionToSliderValues(nextSettings),
      },
      smoothing: smoothingSettings,
      occlusion: nextSettings,
    })
  }

  const resetFaceOcclusion = () => {
    updateFaceOcclusion({ ...DEFAULT_FACE_OCCLUSION_SETTINGS }, 'reset')
  }

  /**
   * Make the automatic anatomical clearance permanent by adding it to this
   * product's calibration Z. The automatic lift then decays to ~0 because the
   * frame is no longer embedded, so the glasses do not visibly move.
   */
  const bakeClearance = (clearanceCm: number) => {
    if (!(clearanceCm > 0)) return
    updateCalibration({ ...calibration, z: Number((calibration.z + clearanceCm).toFixed(3)) }, 'change')
  }

  const exportCalibration = () => {
    const json = JSON.stringify(calibration, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${(products.find((product) => product.id === selectedProductId)?.name ?? 'eyewear').toLowerCase().replace(/[^a-z0-9]+/g, '-')}-calibration.json`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)

    logCalibrationDiagnostics(calibration, {
      source: 'export',
      sliderValues: {
        ...calibrationToSliderValues(calibration),
        ...smoothingToSliderValues(smoothingSettings),
        ...faceOcclusionToSliderValues(faceOcclusionSettings),
      },
      smoothing: smoothingSettings,
      occlusion: faceOcclusionSettings,
    })
  }

  const exportSmoothing = () => {
    const json = JSON.stringify(smoothingSettings, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'motion-smoothing.json'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)

    logCalibrationDiagnostics(calibration, {
      source: 'export',
      sliderValues: {
        ...calibrationToSliderValues(calibration),
        ...smoothingToSliderValues(smoothingSettings),
        ...faceOcclusionToSliderValues(faceOcclusionSettings),
      },
      smoothing: smoothingSettings,
      occlusion: faceOcclusionSettings,
    })
  }

  const exportFaceOcclusion = () => {
    const json = JSON.stringify(faceOcclusionSettings, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'face-occlusion.json'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)

    logCalibrationDiagnostics(calibration, {
      source: 'export',
      sliderValues: {
        ...calibrationToSliderValues(calibration),
        ...smoothingToSliderValues(smoothingSettings),
        ...faceOcclusionToSliderValues(faceOcclusionSettings),
      },
      smoothing: smoothingSettings,
      occlusion: faceOcclusionSettings,
    })
  }

  const selectModelsFolder = async () => {
    modelsFolderRef.current = await chooseLocalModelsFolder()
  }

  const uploadGLB = async (file: File, displayName: string) => {
    if (!modelsFolderRef.current) {
      modelsFolderRef.current = await chooseLocalModelsFolder()
    }

    await saveSelectedGLBToFolder(modelsFolderRef.current, file, displayName)
    uploadSequenceRef.current += 1
    const { product, result } = createUploadedProduct(file, displayName, uploadSequenceRef.current)
    productService.registerUploadedProduct(product)
    uploadedObjectUrlsRef.current.push(result.objectUrl)
    setProductCatalogVersion((version) => version + 1)
    setSelectedProductId(product.id)
    setCalibration(product.calibration)
  }

  const stopCamera = () => {
    stopTrackingPreview()
    latestFacePoseRef.current = null
    arRendererRef.current?.setFacePose(null)
    arRendererRef.current?.setFaceLandmarks(null)
    arRendererRef.current?.setForegroundMask(null)
    meshOverlayRef.current?.clear()
    faceDebugOverlayRef.current?.clear()
    trackingStateRef.current.reset()
    setTrackingState('searching')
    setShowFaceMesh(false)
    setTrackingError('')
    setArError('')
    cameraManagerRef.current?.stop()
    setCameraState('landing')
    setCameraMessage('')
  }

  return (
    <main className="try-on-view">
      <div className="try-on-main">
        <section className={`try-on-surface ${cameraState === 'active' ? 'try-on-surface--camera-active' : ''}`} aria-label="Try-on surface">
        <div className="camera-stage">
          <video
            ref={videoRef}
            className="camera-source"
            playsInline
            muted
            autoPlay
            aria-label="Live camera preview"
          />
          <canvas
            ref={(canvas) => {
              arCanvasRef.current = canvas
            }}
            className="ar-renderer-canvas"
            aria-hidden="true"
          />
          <canvas
            ref={(canvas) => {
              if (canvas && meshOverlayRef.current) {
                meshOverlayRef.current.attach(canvas)
                const rect = canvas.parentElement?.getBoundingClientRect()
                if (rect) meshOverlayRef.current.resize(rect.width, rect.height)
              }
            }}
            className={`face-mesh-overlay ${showFaceMesh ? 'face-mesh-overlay--visible' : ''}`}
            aria-hidden="true"
          />
          <div className="camera-overlay" aria-hidden="true" />
          <pre
            ref={(element) => {
              if (element && faceDebugOverlayRef.current) {
                faceDebugOverlayRef.current.attach(element)
              }
            }}
            className="face-debug-overlay"
            aria-label="Face tracking diagnostics"
          >
            Face: searching
            {'\n'}Pose: unavailable
          </pre>
        </div>

        {arError && cameraState === 'active' && (
          <p className="tracking-error tracking-error--renderer" role="alert">{arError}</p>
        )}

        {trackingError && cameraState === 'active' && (
          <p className="tracking-error" role="alert">{trackingError}</p>
        )}

        <div className="camera-permission" aria-live="polite">
          {cameraState === 'landing' && (
            <div className="camera-permission__content">
              <p className="eyebrow">Camera access</p>
              <h2>Try eyewear on virtually</h2>
              <p>
                Your camera is used in this browser to position the try-on experience. Camera access starts only when you choose to begin.
              </p>
              <button type="button" className="camera-permission__button" onClick={startCamera}>
                Start Try-On
              </button>
            </div>
          )}

          {cameraState === 'requesting' && (
            <div className="camera-permission__content">
              <p className="eyebrow">Camera access</p>
              <h2>Allow camera access</h2>
              <p>{cameraMessage}</p>
            </div>
          )}

          {cameraState === 'active' && (
            <div className="camera-active-controls">
              <span className="camera-active-controls__status">Camera ready</span>
              <span className={`tracking-status tracking-status--${trackingState}`}>Face: {trackingState}</span>
              <button type="button" className="camera-permission__button camera-permission__button--secondary" onClick={toggleFaceMesh}>
                {showFaceMesh ? 'Hide Face Mesh' : 'Show Face Mesh'}
              </button>
              <button type="button" className="camera-permission__button camera-permission__button--secondary" onClick={stopCamera}>
                Stop Camera
              </button>
            </div>
          )}

          {cameraState === 'denied' && (
            <div className="camera-permission__content">
              <p className="eyebrow">Camera permission</p>
              <h2>Camera access is required</h2>
              <p>{cameraMessage}</p>
              <button type="button" className="camera-permission__button" onClick={startCamera}>
                Retry
              </button>
            </div>
          )}

          {cameraState === 'unavailable' && (
            <div className="camera-permission__content">
              <p className="eyebrow">Camera unavailable</p>
              <h2>We could not start the camera</h2>
              <p>{cameraMessage}</p>
              <button type="button" className="camera-permission__button" onClick={startCamera}>
                Retry
              </button>
            </div>
          )}
        </div>
        </section>

        <ProductCarousel
          products={products}
          selectedProductId={selectedProductId}
          loadingProductId={loadingProductId}
          error={arError}
          onSelect={setSelectedProductId}
        />
      </div>

      <section className={`diagnostics-drawer${diagnosticsOpen ? ' diagnostics-drawer--open' : ''}`} aria-label="Developer diagnostics">
        <button
          type="button"
          className="diagnostics-drawer__handle"
          aria-expanded={diagnosticsOpen}
          aria-controls="diagnostics-drawer-content"
          onClick={() => setDiagnosticsOpen((open) => !open)}
        >
          <span className={`diagnostics-drawer__indicator diagnostics-drawer__indicator--${cameraState === 'active' ? trackingState : cameraState}`} />
          <span className="diagnostics-drawer__handle-title">SYSTEM INSIGHT</span>
          <span className="diagnostics-drawer__summary">
            {cameraState === 'active' ? `CAMERA LIVE · FACE ${trackingState.toUpperCase()}` : `CAMERA ${cameraState.toUpperCase()}`}
            {loadingProductId ? ' · LOADING MODEL' : ''}
          </span>
          <span className="diagnostics-drawer__chevron" aria-hidden="true">{diagnosticsOpen ? '⌄' : '⌃'}</span>
        </button>
        <div id="diagnostics-drawer-content" className="diagnostics-drawer__content" aria-hidden={!diagnosticsOpen} inert={!diagnosticsOpen}>
          <div className="diagnostics-drawer__intro">
            <div>
              <p className="eyebrow">Runtime observability</p>
              <h2>Live system diagnostics</h2>
            </div>
            <span className="diagnostics-drawer__live"><i /> {cameraState === 'active' ? 'LIVE' : 'STANDBY'}</span>
          </div>
          <div className="diagnostics-hud" aria-live="polite">
            <div className="diagnostics-hud__metric"><span>Camera</span><strong>{cameraState}</strong><small>{cameraMessage || 'Awaiting user permission'}</small></div>
            <div className="diagnostics-hud__metric"><span>Face tracking</span><strong>{trackingState}</strong><small>{trackingError || (trackingState === 'detected' ? 'Face pose is being tracked locally' : 'Waiting for a face')}</small></div>
            <div className="diagnostics-hud__metric"><span>Active eyewear</span><strong>{products.find((p) => p.id === selectedProductId)?.name ?? 'None'}</strong><small>{loadingProductId ? 'GLB loading in progress' : arError || 'Current catalog model'}</small></div>
            <div className="diagnostics-hud__metric"><span>Calibration</span><strong>{autoCalibrationResult ? (autoCalibrationResult.isAutoApplied ? 'Auto-applied' : 'Catalog calibration') : 'Catalog calibration'}</strong><small>{autoCalibrationResult?.warningMessages.join(' · ') || 'Model-specific calibration active'}</small></div>
            <div className="diagnostics-hud__metric"><span>Face occlusion</span><strong>{faceOcclusionSettings.enabled ? 'Enabled' : 'Disabled'}</strong><small>{occlusionStatus ? `Mode ${occlusionStatus.occlusion.occluder.mode} · segmenter ${occlusionStatus.segmenter}` : 'Runtime status available when camera is active'}</small></div>
            <div className="diagnostics-hud__metric"><span>Pose smoothing</span><strong>{Math.round(smoothingSettings.positionHalfLifeSeconds * 1000)} ms position · {Math.round(smoothingSettings.rotationHalfLifeSeconds * 1000)} ms rotation</strong><small>Configured smoothing half-life values</small></div>
          </div>
          <div className="diagnostics-drawer__actions">
            <button type="button" onClick={exportCalibration}>Save this fit</button>
            <button type="button" onClick={toggleFaceMesh}>{showFaceMesh ? 'Hide face mesh' : 'Developer testing · Face mesh'}</button>
          </div>
        <GLBUploadPanel
          folderSelected={modelsFolderRef.current !== null}
          onChooseFolder={selectModelsFolder}
          onUpload={uploadGLB}
        />
        <GLBDiagnosticPanel
          result={autoCalibrationResult}
          modelUrl={products.find((p) => p.id === selectedProductId)?.model}
          onApplyAutoCalibration={() => {
            if (activeLoadedModelRef.current) {
              const analysis = glbAnalyzer.analyzeAsset(activeLoadedModelRef.current.scene)
              const result = autoCalibrationEngine.computeAutoCalibration(analysis, null)
              setAutoCalibrationResult(result)
              updateCalibration(result.finalCalibration, 'change')
            }
          }}
          onResetManualCorrection={() => {
            if (autoCalibrationResult) {
              updateCalibration(autoCalibrationResult.autoCalibration, 'reset')
            }
          }}
        />
        <CalibrationPanel
          calibration={calibration}
          onChange={updateCalibration}
          onReset={resetCalibration}
          onExport={exportCalibration}
        />
        <SmoothingPanel
          settings={smoothingSettings}
          onChange={updateSmoothingSettings}
          onReset={resetSmoothing}
          onExport={exportSmoothing}
        />
        <FaceOcclusionPanel
          settings={faceOcclusionSettings}
          status={occlusionStatus}
          onBakeClearance={bakeClearance}
          onChange={updateFaceOcclusion}
          onReset={resetFaceOcclusion}
          onExport={exportFaceOcclusion}
        />
        </div>
      </section>
    </main>
  )
}

function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Virtual try-on</p>
          <h1>Eyewear Try-On</h1>
        </div>
        <span className="app-header__status">POC</span>
      </header>
      <TryOnView />
    </div>
  )
}

export default App
