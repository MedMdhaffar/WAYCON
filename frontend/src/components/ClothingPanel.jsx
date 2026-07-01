import { useState, useEffect } from 'react'

const FIELDS = ['top', 'bottom', 'shoes', 'full']

export default function ClothingPanel({ bestBodyCrops, clothingStructured, onChange }) {
  const [values, setValues] = useState({ top: '', bottom: '', shoes: '', full: '' })

  useEffect(() => {
    if (clothingStructured && Object.keys(clothingStructured).length) {
      setValues({
        top: clothingStructured.top ?? '',
        bottom: clothingStructured.bottom ?? '',
        shoes: clothingStructured.shoes ?? '',
        full: clothingStructured.full ?? '',
      })
    }
  }, [clothingStructured])

  const update = (field, val) => {
    const next = { ...values, [field]: val }
    setValues(next)
    onChange?.(next)
  }

  const inputStyle = {
    width: '100%', padding: '7px 11px', background: '#0f1117',
    border: '1px solid #1e2330', borderRadius: '6px',
    color: '#e2e8f0', fontSize: '14px',
  }
  const labelStyle = { fontSize: '12px', color: '#94a3b8', marginBottom: '5px', display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }

  return (
    <div className="card">
      <div className="card-title">Clothing Description</div>

      {bestBodyCrops.length > 0 && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', overflowX: 'auto', paddingBottom: '4px' }}>
          {bestBodyCrops.map((p, i) => (
            <img
              key={i}
              src={`/api/images?path=${encodeURIComponent(p)}`}
              alt={`best ${i + 1}`}
              style={{ height: '120px', width: 'auto', borderRadius: '6px', border: '1px solid #1e2330', flexShrink: 0 }}
            />
          ))}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        {FIELDS.map(f => (
          <div key={f} style={f === 'full' ? { gridColumn: '1 / -1' } : {}}>
            <label style={labelStyle}>{f}</label>
            <input style={inputStyle} value={values[f]} onChange={e => update(f, e.target.value)} placeholder={`Describe ${f}…`} />
          </div>
        ))}
      </div>
    </div>
  )
}
