// 타일링을 모달로 감싼다. 계산과 경고는 전부 `TilingCard` 에 있다 — 여기서는
// 열고 닫기만 한다.
import { Modal } from '@mantine/core'
import type { DatasetOut } from '../../api/client'
import TilingCard from './TilingCard'

interface Props {
  projectId: string
  datasetId: string
  dataset: DatasetOut
  opened: boolean
  onClose: () => void
}

export default function TilingModal({ projectId, datasetId, dataset, opened, onClose }: Props) {
  return (
    <Modal opened={opened} onClose={onClose} title="Tile into training patches" size="lg">
      <TilingCard
        projectId={projectId}
        datasetId={datasetId}
        dataset={dataset}
        onStarted={onClose}
      />
    </Modal>
  )
}
