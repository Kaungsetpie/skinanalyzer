import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Image, Pressable, Text, View } from "react-native";
import type { FeatureItem } from "../types/data";

type FeatureCardProps = {
  feature: FeatureItem;
  trustedFooter?: {
    label: string;
    avatars: string[];
  };
  onDemoPress?: () => void;
};

function iconFor(name: string): keyof typeof MaterialCommunityIcons.glyphMap {
  const map: Record<string, keyof typeof MaterialCommunityIcons.glyphMap> = {
    microscope: "microscope",
    camera: "camera",
    water: "water",
    "shield-check": "shield-check",
    "chart-timeline-variant": "chart-timeline-variant",
  };
  return map[name] ?? "star-four-points-outline";
}

export function FeatureCard({
  feature,
  trustedFooter,
  onDemoPress,
}: FeatureCardProps) {
  const icon = iconFor(feature.iconName);
  const isPrimary = feature.variant === "primary";
  const isMuted = feature.variant === "muted";

  const container = isPrimary
    ? "bg-teal-alt"
    : isMuted
      ? "bg-slate-100"
      : "bg-white border border-slate-100";

  const titleColor = isPrimary ? "text-white" : "text-slate-900";
  const bodyColor = isPrimary ? "text-white/90" : "text-slate-600";
  const iconWrap = isPrimary
    ? "bg-white/15"
    : "bg-[#E0F2F1]";
  const iconColor = isPrimary ? "#ffffff" : "#006D77";

  return (
    <View className={`rounded-3xl p-5 shadow-sm ${container}`}>
      <View className="flex-row items-start gap-4">
        <View
          className={`h-12 w-12 items-center justify-center rounded-2xl ${iconWrap}`}
        >
          <MaterialCommunityIcons name={icon} size={26} color={iconColor} />
        </View>
        <View className="flex-1">
          <Text className={`text-lg font-bold ${titleColor}`}>{feature.title}</Text>
          <Text className={`mt-1 text-sm leading-6 ${bodyColor}`}>
            {feature.description}
          </Text>
        </View>
      </View>

      {feature.footerTrusted && trustedFooter ? (
        <View className="mt-4 flex-row items-center gap-3 border-t border-slate-200/60 pt-4">
          <View className="flex-row">
            {trustedFooter.avatars.map((uri, i) => (
              <Image
                key={uri}
                source={{ uri }}
                className={`h-8 w-8 rounded-full border-2 border-white ${i === 0 ? "" : "-ml-2"}`}
                style={{ zIndex: trustedFooter.avatars.length - i }}
              />
            ))}
          </View>
          <Text className="text-xs font-medium text-slate-600">
            {trustedFooter.label}
          </Text>
        </View>
      ) : null}

      {feature.demoLabel && isPrimary ? (
        <Pressable
          onPress={onDemoPress}
          className="mt-4 self-start rounded-full border border-white/50 px-4 py-2 active:bg-white/10"
        >
          <Text className="text-sm font-semibold text-white">
            {feature.demoLabel}
          </Text>
        </Pressable>
      ) : null}
    </View>
  );
}
