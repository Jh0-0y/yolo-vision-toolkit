// 연구실 크롭의 검출 설정 — 공용 검출기 목록 + **영상 샘플링 간격.**
//
// 엔트리 UI 는 학습실 오토라벨링과 같은 것을 쓴다(`components/detect/DetectorList`).
// 여기 남는 것은 연구실에만 있는 것뿐이다: 영상은 프레임을 얼마나 자주 볼지 정해야 한다.
import { NumberInput } from '@mantine/core'
import type { ModelOut } from '../../api/client'
import DetectorList, { type DetectorEntry } from '../detect/DetectorList'

export { IMGSZ_OPTIONS, newEntry } from '../detect/DetectorList'
export type { DetectorEntry } from '../detect/DetectorList'

interface Props {
  models: ModelOut[]
  entries: DetectorEntry[]
  onEntries: (v: DetectorEntry[]) => void
  sampling: number | ''
  onSampling: (v: number | '') => void
  disabled?: boolean
}

export default function DetectionSettings({
  models,
  entries,
  onEntries,
  sampling,
  onSampling,
  disabled,
}: Props) {
  return (
    <>
      <DetectorList
        models={models}
        entries={entries}
        onEntries={onEntries}
        disabled={disabled}
      />

      <NumberInput
        label="Sampling interval (ms)"
        placeholder="default 100"
        value={sampling}
        onChange={(v) => onSampling(v === '' || v == null ? '' : Number(v))}
        min={10}
        step={10}
        disabled={disabled}
        maw={260}
      />
    </>
  )
}
