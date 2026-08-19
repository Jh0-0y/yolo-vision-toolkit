// 나누기를 모달로 감싼다. 계산과 경고는 전부 `SplitCard` 에 있다 — 여기서는
// 언제 보일지만 정한다.
import { Modal } from '@mantine/core'
import SplitCard from './SplitCard'
import type { DatasetOut } from '../../api/client'

interface Props {
  projectId: string
  datasetId: string
  dataset: DatasetOut
  opened: boolean
  onClose: () => void
}

export default function SplitModal({ projectId, datasetId, dataset, opened, onClose }: Props) {
  return (
    <Modal opened={opened} onClose={onClose} title="Split into train / val / test" size="lg">
      <SplitCard projectId={projectId} datasetId={datasetId} dataset={dataset} />
    </Modal>
  )
}
