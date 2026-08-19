// 서버 통신의 단일 출입구.
//
// 실제 구현은 리소스별 파일에 있고 여기서 전부 재수출한다 — 화면은 예나 지금이나
// `from '../api/client'` 하나만 알면 된다. 새 경로를 추가할 때는 해당 리소스
// 파일에 넣고, 새 리소스라면 파일을 만들어 아래에 한 줄 더한다.

export * from './http'
export * from './system'
export * from './models'
export * from './projects'
export * from './datasets'
export * from './labs'
export * from './labCrops'
export * from './labels'
export * from './classes'
export * from './jobs'
export * from './training'
export * from './test/predict'
export * from './test/compare'
