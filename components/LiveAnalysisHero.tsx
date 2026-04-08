import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Image, Text, View } from "react-native";

type LiveAnalysisHeroProps = {
  imageUri: string;
  tag: string;
  title: string;
  scanningLabel: string;
  hydrationLabel: string;
  hydrationPercent: number;
};

export function LiveAnalysisHero({
  imageUri,
  tag,
  title,
  scanningLabel,
  hydrationLabel,
  hydrationPercent,
}: LiveAnalysisHeroProps) {
  const progress = Math.min(100, Math.max(0, hydrationPercent)) / 100;

  return (
    <View className="overflow-hidden rounded-3xl bg-slate-900 shadow-md">
      <View className="relative aspect-[4/5] w-full">
        <Image source={{ uri: imageUri }} className="h-full w-full" resizeMode="cover" />
        <View className="absolute left-4 top-4 rounded-full bg-white/90 px-3 py-1">
          <Text className="text-xs font-bold tracking-wide text-slate-800">
            {tag}
          </Text>
        </View>
        <View className="absolute bottom-0 left-0 right-0 bg-black/45 px-4 pb-5 pt-4">
          <View className="mb-2 flex-row items-center justify-between">
            <Text className="text-lg font-bold text-white">{title}</Text>
            <View className="flex-row items-center gap-1.5 rounded-full bg-red-500/90 px-2 py-1">
              <View className="h-2 w-2 rounded-full bg-white" />
              <Text className="text-[10px] font-bold tracking-wide text-white">
                {scanningLabel}
              </Text>
            </View>
          </View>
          <View className="h-2 w-full overflow-hidden rounded-full bg-white/25">
            <View
              className="h-full rounded-full bg-teal-alt"
              style={{ width: `${progress * 100}%` }}
            />
          </View>
          <Text className="mt-2 text-sm text-white/90">
            {hydrationLabel}: {hydrationPercent}%
          </Text>
        </View>
      </View>
    </View>
  );
}
