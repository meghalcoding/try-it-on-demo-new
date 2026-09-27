import { useEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react'
import type { Product } from '../types/Product'

export interface ProductCarouselProps {
  readonly products: readonly Product[]
  readonly selectedProductId: string
  readonly loadingProductId: string | null
  readonly error: string
  readonly onSelect: (productId: string) => void
}

const SWIPE_THRESHOLD_PX = 28

export function ProductCarousel({
  products,
  selectedProductId,
  loadingProductId,
  error,
  onSelect,
}: ProductCarouselProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const itemRefs = useRef(new Map<string, HTMLButtonElement>())
  const pointerStartXRef = useRef<number | null>(null)
  const pointerLastXRef = useRef<number | null>(null)
  const suppressClickRef = useRef(false)

  const selectedIndex = Math.max(
    0,
    products.findIndex((product) => product.id === selectedProductId),
  )

  useEffect(() => {
    const selectedItem = itemRefs.current.get(selectedProductId)
    selectedItem?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
  }, [selectedProductId])

  const selectByOffset = (offset: number) => {
    if (products.length === 0) return
    const nextIndex = Math.min(products.length - 1, Math.max(0, selectedIndex + offset))
    onSelect(products[nextIndex].id)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault()
      selectByOffset(1)
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault()
      selectByOffset(-1)
    } else if (event.key === 'Home') {
      event.preventDefault()
      products[0] && onSelect(products[0].id)
    } else if (event.key === 'End') {
      event.preventDefault()
      products[products.length - 1] && onSelect(products[products.length - 1].id)
    }
  }

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return
    pointerStartXRef.current = event.clientX
    pointerLastXRef.current = event.clientX
    suppressClickRef.current = false
    viewportRef.current?.setPointerCapture(event.pointerId)
  }

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (pointerStartXRef.current === null) return
    pointerLastXRef.current = event.clientX
    const distance = Math.abs(event.clientX - pointerStartXRef.current)
    if (distance >= SWIPE_THRESHOLD_PX) {
      suppressClickRef.current = true
    }
  }

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (pointerStartXRef.current === null || pointerLastXRef.current === null) return
    const deltaX = pointerLastXRef.current - pointerStartXRef.current
    const wasSwipe = Math.abs(deltaX) >= SWIPE_THRESHOLD_PX

    if (wasSwipe) {
      selectByOffset(deltaX < 0 ? 1 : -1)
    }

    pointerStartXRef.current = null
    pointerLastXRef.current = null
    viewportRef.current?.releasePointerCapture(event.pointerId)
  }

  const handleClick = (productId: string) => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false
      return
    }
    onSelect(productId)
  }

  return (
    <section className="product-carousel" aria-label="Eyewear products">
      <div className="product-carousel__header">
        <div>
          <p className="eyebrow">Eyewear</p>
          <h2>Select a frame</h2>
        </div>
        <span className="product-carousel__count" aria-live="polite">
          {selectedIndex + 1} / {products.length}
        </span>
      </div>

      <div className="product-carousel__body">
        <button
          type="button"
          className="product-carousel__arrow"
          onClick={() => selectByOffset(-1)}
          disabled={selectedIndex <= 0}
          aria-label="Previous eyewear"
        >
          ‹
        </button>

        <div
          ref={viewportRef}
          className="product-carousel__viewport"
          tabIndex={0}
          role="listbox"
          aria-label="Available eyewear frames"
          aria-activedescendant={`product-option-${selectedProductId}`}
          onKeyDown={handleKeyDown}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          <div className="product-carousel__track">
            {products.map((product) => {
              const selected = product.id === selectedProductId
              const loading = product.id === loadingProductId

              return (
                <button
                  key={product.id}
                  id={`product-option-${product.id}`}
                  ref={(element) => {
                    if (element) itemRefs.current.set(product.id, element)
                    else itemRefs.current.delete(product.id)
                  }}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  aria-label={`${product.name}${selected ? ', selected' : ''}${loading ? ', loading' : ''}`}
                  className={`product-card${selected ? ' product-card--selected' : ''}`}
                  onClick={() => handleClick(product.id)}
                >
                  <span className="product-card__thumbnail-wrap">
                    <img className="product-card__thumbnail" src={product.thumbnail} alt="" draggable={false} />
                    {selected && <span className="product-card__selected-badge">Selected</span>}
                    {loading && <span className="product-card__loading-badge">Loading…</span>}
                  </span>
                  <span className="product-card__name">{product.name}</span>
                  <span className="product-card__meta">{product.color}</span>
                </button>
              )
            })}
          </div>
        </div>

        <button
          type="button"
          className="product-carousel__arrow"
          onClick={() => selectByOffset(1)}
          disabled={selectedIndex >= products.length - 1}
          aria-label="Next eyewear"
        >
          ›
        </button>
      </div>

      {error && (
        <p className="product-carousel__error" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
