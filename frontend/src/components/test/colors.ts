// Class → color, stable by class id (shared across Test views).
const PALETTE = [
  '#4dabf7', '#51cf66', '#ff6b6b', '#ffd43b', '#cc5de8',
  '#20c997', '#ff922b', '#f783ac', '#748ffc', '#a9e34b',
]

export const clsColor = (cls: number) =>
  PALETTE[((cls % PALETTE.length) + PALETTE.length) % PALETTE.length]
