// 카드 격자 — 큰 차트 하나가 아니라 같은 크기의 카드를 여러 장 깔고 한눈에 훑는다.
import {
  SimpleGrid,
} from '@mantine/core'


export default function ChartGrid({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <SimpleGrid cols={{ base: 1, md: 2, xl: 3 }} spacing="md">
      {children}
    </SimpleGrid>
  )
}
