import { Text, View } from "react-native";
import Svg, { Circle } from "react-native-svg";

type DonutScoreProps = {
  percent: number;
  label: string;
  size?: number;
  stroke?: number;
};

export function DonutScore({
  percent,
  label,
  size = 140,
  stroke = 12,
}: DonutScoreProps) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.min(100, Math.max(0, percent));
  const dash = (clamped / 100) * c;

  return (
    <View className="items-center justify-center" style={{ width: size, height: size }}>
      <Svg width={size} height={size}>
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="#BFDBFE"
          strokeWidth={stroke}
          fill="none"
        />
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="#004D4D"
          strokeWidth={stroke}
          fill="none"
          strokeDasharray={`${dash} ${c - dash}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </Svg>
      <View className="absolute items-center justify-center">
        <Text className="text-2xl font-extrabold text-teal-dark">{clamped}%</Text>
        <Text className="text-xs font-medium text-slate-500">{label}</Text>
      </View>
    </View>
  );
}
