import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Text, View } from "react-native";

type Variant = "teal" | "blue";

type InfoCardProps = {
  icon: keyof typeof MaterialCommunityIcons.glyphMap;
  title: string;
  description: string;
  variant?: Variant;
};

const styles: Record<
  Variant,
  { wrap: string; iconBg: string; iconColor: string }
> = {
  teal: {
    wrap: "bg-[#E0F7F8]",
    iconBg: "bg-white/80",
    iconColor: "#00797C",
  },
  blue: {
    wrap: "bg-[#D1E9FF]",
    iconBg: "bg-white/80",
    iconColor: "#0369a1",
  },
};

export function InfoCard({
  icon,
  title,
  description,
  variant = "teal",
}: InfoCardProps) {
  const s = styles[variant];

  return (
    <View className={`flex-1 rounded-2xl p-4 shadow-sm ${s.wrap}`}>
      <View
        className={`mb-3 h-10 w-10 items-center justify-center rounded-full ${s.iconBg}`}
      >
        <MaterialCommunityIcons name={icon} size={22} color={s.iconColor} />
      </View>
      <Text className="text-base font-bold text-slate-900">{title}</Text>
      <Text className="mt-1 text-xs leading-5 text-slate-600">{description}</Text>
    </View>
  );
}
