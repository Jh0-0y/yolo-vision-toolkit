// 홈 — 두 공간의 갈림길. **여기서는 목록을 보여주지 않는다.**
//
// 축이 다르다. 도메인(농구·야구)은 프로젝트가, 클래스 범위(공·선수)는 데이터셋이,
// 연구는 별도 공간이 갖는다. 두 공간은 **모델 풀만 공유**하고 영상은 각자 갖는다 —
// 학습용 영상(장면 샘플)과 연구용 영상(크롭할 경기)은 실제로 다른 영상이다.
//
// 목록은 각 공간에 들어가서 본다. 홈이 하는 일은 갈림길을 보여주는 것뿐이다.
import {
  Card,
  Container,
  Group,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
  UnstyledButton,
} from '@mantine/core'
import { IconChevronRight, IconFlask, IconSchool, IconTool } from '@tabler/icons-react'
import { Link } from 'react-router-dom'

interface DoorProps {
  to: string
  icon: typeof IconFlask
  color: string
  title: string
  subtitle: string
}

function Door({ to, icon: Icon, color, title, subtitle }: DoorProps) {
  return (
    <UnstyledButton component={Link} to={to}>
      <Card withBorder radius="md" padding="xl" h="100%">
        <Group justify="space-between" wrap="nowrap" align="flex-start">
          <Group gap="md" wrap="nowrap">
            <ThemeIcon size={56} radius="md" variant="light" color={color}>
              <Icon size={32} stroke={1.6} />
            </ThemeIcon>
            <div style={{ minWidth: 0 }}>
              <Title order={3}>{title}</Title>
              <Text c="dimmed" size="sm">
                {subtitle}
              </Text>
            </div>
          </Group>
          <IconChevronRight size={20} stroke={1.6} style={{ flexShrink: 0, marginTop: 8 }} />
        </Group>
      </Card>
    </UnstyledButton>
  )
}

export default function HomePage() {
  return (
    <Container size={1100} py="xl">
      <Stack gap="xl">
        <Group gap="md" pt="md">
          <ThemeIcon size={52} radius="md" variant="light">
            <IconTool size={30} stroke={1.6} />
          </ThemeIcon>
          <div>
            <Title order={2}>YOLO Vision Toolkit</Title>
            <Text c="dimmed" size="sm">
              Pick a space — research crops in the Lab, build datasets and models in the Studio.
            </Text>
          </div>
        </Group>

        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
          <Door
            to="/lab"
            icon={IconFlask}
            color="grape"
            title="Lab"
            subtitle="Crop research — videos, runs and their outputs"
          />
          <Door
            to="/studio"
            icon={IconSchool}
            color="blue"
            title="Studio"
            subtitle="Datasets, labeling and training — one project per domain"
          />
        </SimpleGrid>
      </Stack>
    </Container>
  )
}
