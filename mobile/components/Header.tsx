import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Image, Pressable, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

type HeaderProps = {
  title: string;
  avatarUri: string;
  variant?: "menu" | "back";
  onMenuPress?: () => void;
  onBackPress?: () => void;
  onAvatarPress?: () => void;
};

export function Header({
  title,
  avatarUri,
  variant = "menu",
  onMenuPress,
  onBackPress,
  onAvatarPress,
}: HeaderProps) {
  const insets = useSafeAreaInsets();
  const isBack = variant === "back";

  return (
    <View
      className="flex-row items-center justify-between border-b border-slate-100 bg-white px-5 pb-3"
      style={{ paddingTop: Math.max(insets.top, 12) }}
    >
      <Pressable
        onPress={isBack ? onBackPress : onMenuPress}
        className="h-10 w-10 items-center justify-center rounded-full active:bg-slate-100"
        accessibilityRole="button"
        accessibilityLabel={isBack ? "Go back" : "Open menu"}
      >
        <MaterialCommunityIcons
          name={isBack ? "arrow-left" : "menu"}
          size={24}
          color="#0f172a"
        />
      </Pressable>
      <Text className="text-lg font-bold text-teal-dark">{title}</Text>
      <Pressable
        onPress={onAvatarPress}
        className="h-10 w-10 overflow-hidden rounded-full border border-slate-200 active:opacity-80"
        accessibilityRole="button"
        accessibilityLabel="Profile"
      >
        <Image source={{ uri: avatarUri }} className="h-full w-full" />
      </Pressable>
    </View>
  );
}
