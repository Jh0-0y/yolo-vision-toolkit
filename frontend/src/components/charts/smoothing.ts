// 텐서보드의 스무딩 슬라이더가 쓰는 그 계산. 차트 조각들과 함께 두어
// 벤치마크든 학습이든 같은 곡선 보정을 쓰게 한다.

/** 텐서보드와 같은 지수이동평균. `weight` 0 이면 원본 그대로, 1 에 가까울수록 평평해진다.
 *
 *  단순 EMA 는 **앞부분이 0 쪽으로 끌려간다** — 첫 값을 0 에서 시작한 것처럼 보이게 하기
 *  때문이다. 그래서 텐서보드는 `1 - weight^(i+1)` 로 나눠 그 편향을 걷어낸다(debias).
 *  그 보정을 빼면 학습 초반 몇 에폭이 실제보다 낮게 그려진다.
 */
export function smoothSeries(
  values: (number | undefined)[],
  weight: number,
): (number | undefined)[] {
  if (weight <= 0) return values
  const out: (number | undefined)[] = []
  let last = 0
  let seen = 0
  for (const v of values) {
    if (v === undefined || Number.isNaN(v)) {
      out.push(undefined) // 구멍은 구멍으로 둔다 — 없는 에폭을 지어내지 않는다
      continue
    }
    last = last * weight + (1 - weight) * v
    seen += 1
    out.push(last / (1 - weight ** seen))
  }
  return out
}
