import rawProducts from '../data/products.json'
import type { Calibration } from '../types/Calibration'
import type { Product } from '../types/Product'



const CALIBRATION_KEYS: readonly (keyof Calibration)[] = [
  'scale',
  'x',
  'y',
  'z',
  'rotationX',
  'rotationY',
  'rotationZ',
]

function isCalibration(value: unknown): value is Calibration {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return CALIBRATION_KEYS.every((key) => typeof record[key] === 'number' && Number.isFinite(record[key] as number))
}

function isProduct(value: unknown): value is Product {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.id === 'string' && record.id.trim().length > 0 &&
    typeof record.name === 'string' && record.name.trim().length > 0 &&
    typeof record.price === 'number' && Number.isFinite(record.price) &&
    typeof record.currency === 'string' && record.currency.trim().length > 0 &&
    typeof record.thumbnail === 'string' && record.thumbnail.trim().length > 0 &&
    typeof record.model === 'string' && record.model.trim().length > 0 &&
    typeof record.color === 'string' && record.color.trim().length > 0 &&
    typeof record.size === 'string' && record.size.trim().length > 0 &&
    isCalibration(record.calibration)
  )
}

const products = Object.freeze(
  rawProducts.map((product, index) => {
    if (!isProduct(product)) {
      throw new Error(`Invalid product catalog record at index ${index}.`)
    }
    return Object.freeze({
      ...product,
      calibration: Object.freeze({ ...product.calibration }),
    })
  }),
)

export class ProductService {
  private readonly uploadedProducts = new Map<string, Product>()

  listProducts(): readonly Product[] {
    return [...products, ...this.uploadedProducts.values()]
  }

  registerUploadedProduct(product: Product): void {
    if (this.uploadedProducts.has(product.id)) {
      throw new Error(`Uploaded product '${product.id}' is already registered.`)
    }
    this.uploadedProducts.set(product.id, product)
  }

  getProductById(productId: string): Product {
    const product = products.find((candidate) => candidate.id === productId) ?? this.uploadedProducts.get(productId)
    if (!product) {
      throw new Error(`Product '${productId}' was not found in the catalog.`)
    }
    return product
  }
}
