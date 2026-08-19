// 내보내기를 모달로 감싼다. 만든 zip 을 받는 버튼이 `ExportCard` 안에 뜨므로
// **저절로 닫지 않는다** — 닫으면 결과로 가는 손잡이가 사라진다.
import { Modal } from '@mantine/core'
import ExportCard from './ExportCard'
import type { DatasetOut } from '../../api/client'

interface Props {
  projectId: string
  datasetId: string
  dataset: DatasetOut
  opened: boolean
  onClose: () => void
}

export default function ExportModal({ projectId, datasetId, dataset, opened, onClose }: Props) {
  return (
    <Modal opened={opened} onClose={onClose} title="Export" size="lg">
      <ExportCard projectId={projectId} datasetId={datasetId} dataset={dataset} />
    </Modal>
  )
}
