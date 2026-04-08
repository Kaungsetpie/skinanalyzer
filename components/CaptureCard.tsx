import { MaterialCommunityIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { Pressable, Text, View } from "react-native";

type CaptureCardProps = {
  liveStatusLabel: string;
  liveStatusDetail: string;
  captureTitle: string;
  captureHint: string;
  onGalleryPress?: () => void;
  onShutterPress?: () => void;
  onFlipPress?: () => void;
};

export function CaptureCard({
  liveStatusLabel,
  liveStatusDetail,
  captureTitle,
  captureHint,
  onGalleryPress,
  onShutterPress,
  onFlipPress,
}: CaptureCardProps) {
  return (
    <View className="overflow-hidden rounded-3xl shadow-md">
      <LinearGradient
        colors={["#F5F0E8", "#E3F2FD"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={{ padding: 20 }}
      >
        <View className="mb-6 rounded-2xl bg-white/95 p-4 shadow-sm">
          <View className="flex-row items-center gap-3">
            <View className="h-11 w-11 items-center justify-center rounded-full bg-[#E0F7F8]">
              <MaterialCommunityIcons
                name="microscope"
                size={22}
                color="#00797C"
              />
            </View>
            <View className="flex-1">
              <Text className="text-xs font-semibold uppercase tracking-wide text-teal-primary">
                {liveStatusLabel}
              </Text>
              <Text className="mt-0.5 text-sm font-bold text-slate-900">
                {liveStatusDetail}
              </Text>
            </View>
          </View>
        </View>

        <View className="items-center">
          <View className="mb-4 h-36 w-36 items-center justify-center rounded-full bg-[#E0F7F8]">
            <MaterialCommunityIcons name="camera" size={56} color="#00797C" />
          </View>
          <Text className="text-lg font-bold text-slate-900">{captureTitle}</Text>
          <Text className="mt-2 max-w-xs text-center text-sm leading-6 text-slate-500">
            {captureHint}
          </Text>
        </View>

        <View className="mt-8 flex-row items-center justify-center gap-6">
          <Pressable
            onPress={onGalleryPress}
            className="h-12 w-12 items-center justify-center rounded-xl bg-white shadow-sm active:opacity-80"
            accessibilityRole="button"
            accessibilityLabel="Open gallery"
          >
            <MaterialCommunityIcons name="image-outline" size={22} color="#0f172a" />
          </Pressable>
          <Pressable
            onPress={onShutterPress}
            className="h-16 w-16 items-center justify-center rounded-full bg-teal-primary shadow-lg active:opacity-90"
            accessibilityRole="button"
            accessibilityLabel="Take photo"
          >
            <MaterialCommunityIcons name="camera-iris" size={32} color="#ffffff" />
          </Pressable>
          <Pressable
            onPress={onFlipPress}
            className="h-12 w-12 items-center justify-center rounded-xl bg-white shadow-sm active:opacity-80"
            accessibilityRole="button"
            accessibilityLabel="Flip camera"
          >
            <MaterialCommunityIcons name="camera-flip-outline" size={22} color="#0f172a" />
          </Pressable>
        </View>
      </LinearGradient>
    </View>
  );
}
