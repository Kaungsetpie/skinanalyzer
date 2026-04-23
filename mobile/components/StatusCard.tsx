import { Text, View } from "react-native";
import { ActionButton } from "./ActionButton";

type StatusCardProps = {
  title: string;
  description: string;
  ctaLabel: string;
  onCtaPress?: () => void;
};

export function StatusCard({
  title,
  description,
  ctaLabel,
  onCtaPress,
}: StatusCardProps) {
  return (
    <View className="rounded-3xl bg-white p-5 shadow-sm">
      <View className="mb-3 flex-row items-center justify-between">
        <Text className="text-base font-bold text-slate-900">{title}</Text>
        <View className="flex-row gap-1">
          {[0, 1, 2].map((i) => (
            <View
              key={i}
              className="h-2 w-2 rounded-full bg-teal-primary opacity-70"
            />
          ))}
        </View>
      </View>
      <Text className="mb-4 text-sm leading-6 text-slate-600">{description}</Text>
      <ActionButton
        label={ctaLabel}
        onPress={onCtaPress}
        showArrow
        colorClass="bg-teal-primary"
      />
    </View>
  );
}
