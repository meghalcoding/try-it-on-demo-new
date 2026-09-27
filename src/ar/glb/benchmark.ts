import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { GLBAssetAnalyzer } from './GLBAssetAnalyzer'
import { AutoCalibrationEngine } from './AutoCalibrationEngine'

export interface BenchmarkReport {
  readonly modelName: string
  readonly totalVertices: number
  readonly totalTriangles: number
  readonly nativeDimensions: { x: number; y: number; z: number }
  readonly widthAxis: string
  readonly heightAxis: string
  readonly depthAxis: string
  readonly detectedParts: {
    lensesCount: number
    hasBridge: boolean
    templesCount: number
  }
  readonly autoScale: number
  readonly confidenceScore: number
  readonly confidenceTier: string
}

/**
 * Runs the automated GLB geometry analysis & calibration engine against a list of eyewear GLB URLs,
 * returning detailed benchmark metrics for every asset in the corpus.
 */
export async function runCatalogBenchmark(
  modelUrls: string[],
  loader: GLTFLoader = new GLTFLoader(),
): Promise<readonly BenchmarkReport[]> {
  const analyzer = new GLBAssetAnalyzer()
  const engine = new AutoCalibrationEngine()
  const reports: BenchmarkReport[] = []

  for (const url of modelUrls) {
    try {
      const gltf = await loader.loadAsync(url)
      const analysis = analyzer.analyzeAsset(gltf.scene)
      const autoResult = engine.computeAutoCalibration(analysis, null)

      reports.push({
        modelName: url.split('/').pop() || url,
        totalVertices: analysis.summary.totalVertices,
        totalTriangles: analysis.summary.totalTriangles,
        nativeDimensions: {
          x: Number(analysis.summary.overallDimensions.x.toFixed(2)),
          y: Number(analysis.summary.overallDimensions.y.toFixed(2)),
          z: Number(analysis.summary.overallDimensions.z.toFixed(2)),
        },
        widthAxis: analysis.orientation.widthAxis,
        heightAxis: analysis.orientation.heightAxis,
        depthAxis: analysis.orientation.depthAxis,
        detectedParts: {
          lensesCount: analysis.parts.detectedLenses.length,
          hasBridge: analysis.parts.detectedBridge !== null,
          templesCount: analysis.parts.detectedTemples.length,
        },
        autoScale: Number(autoResult.autoCalibration.scale.toFixed(4)),
        confidenceScore: autoResult.confidence.overall,
        confidenceTier: autoResult.confidence.tier,
      })
    } catch (err) {
      console.error(`Benchmark failed for ${url}:`, err)
    }
  }

  return reports
}
