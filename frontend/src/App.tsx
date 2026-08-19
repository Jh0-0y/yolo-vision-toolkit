import { Navigate, Route, Routes } from 'react-router-dom'
import JobIndicator from './components/JobIndicator'
import LabLayout from './layouts/LabLayout'
import ProjectLayout from './layouts/ProjectLayout'
import DatasetDetailPage from './pages/DatasetDetailPage'
import DatasetsPage from './pages/DatasetsPage'
import HomePage from './pages/HomePage'
import LabCropPage from './pages/LabCropPage'
import LabHomePage from './pages/LabHomePage'
import LabCropRunDetailPage from './pages/LabCropRunDetailPage'
import LabCropRunsPage from './pages/LabCropRunsPage'
import LabelEditorPage from './pages/LabelEditorPage'
import ModelsPage from './pages/ModelsPage'
import StudioHomePage from './pages/StudioHomePage'
import TrainPage from './pages/TrainPage'
import TrainingHistoryPage from './pages/TrainingHistoryPage'
import TrainRunDetailPage from './pages/TrainRunDetailPage'

export default function App() {
  return (
    <>
      <JobIndicator />
      <Routes>
        {/* 홈에서 두 공간이 갈린다 — 연구실(크롭)과 학습실(데이터셋·학습) */}
        <Route path="/" element={<HomePage />} />
        <Route path="/studio" element={<StudioHomePage />} />
        {/* /lab 은 목록(무슨 연구인지), 그 아래가 연구실 안이다.
            연구실은 하나뿐이라 경로에 id 가 없다. */}
        <Route path="/lab" element={<LabHomePage />} />
        <Route path="/lab" element={<LabLayout />}>
          <Route path="crop" element={<LabCropPage />} />
          <Route path="crops" element={<LabCropRunsPage />} />
          <Route path="crops/:cropId" element={<LabCropRunDetailPage />} />
        </Route>
      <Route path="/projects/:projectId" element={<ProjectLayout />}>
        <Route index element={<Navigate to="datasets" replace />} />
        {/* 데이터셋이 작업 단위다 — 가져오기·검수·분할·내보내기가 전부 그 안에 있다 */}
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="datasets/:datasetId" element={<DatasetDetailPage />} />
        <Route path="datasets/:datasetId/label/:stem" element={<LabelEditorPage />} />
        {/* 학습은 이력이 먼저 — 새 학습 시작 버튼으로 시작 화면에 들어간다 */}
        <Route path="training" element={<TrainingHistoryPage />} />
        <Route path="training/new" element={<TrainPage />} />
        <Route path="training/:runId" element={<TrainRunDetailPage />} />
        <Route path="models" element={<ModelsPage />} />
        {/* 옛 화면들 — 데이터셋 안으로 들어갔거나 이름이 바뀌었다 */}
        <Route path="dataset" element={<Navigate to="../datasets" replace />} />
        <Route path="dataset/label/:stem" element={<Navigate to="../datasets" replace />} />
        <Route path="upload" element={<Navigate to="../datasets" replace />} />
        <Route path="classes" element={<Navigate to="../datasets" replace />} />
        <Route path="exports" element={<Navigate to="../datasets" replace />} />
        <Route path="train" element={<Navigate to="../training/new" replace />} />
        <Route path="history" element={<Navigate to="../training" replace />} />
        {/* 크롭은 연구실로 옮겨 갔다 — 옛 링크는 연구실 목록으로 보낸다
            (어느 연구실인지는 프로젝트가 알 수 없으므로 사용자가 고른다) */}
        <Route path="lab/*" element={<Navigate to="/lab" replace />} />
        <Route path="test" element={<Navigate to="/lab" replace />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
