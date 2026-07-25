import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Pressable, Text, View } from "react-native";

type ActionButtonProps = {
  label: string;
  onPress?: () => void;
  showArrow?: boolean;
  variant?: "solid" | "outline";
  colorClass?: string;
};

export function ActionButton({
  label,
  onPress,
  showArrow,
  variant = "solid",
  colorClass = "bg-teal-primary",
}: ActionButtonProps) {
  const isOutline = variant === "outline";

  return (
    <Pressable
      onPress={onPress}
      className={`flex-row items-center justify-center rounded-full px-5 py-3.5 active:opacity-90 ${
        isOutline
          ? "border-2 border-teal-dark bg-white"
          : `${colorClass}`
      }`}
    >
      <Text
        className={`text-center text-base font-semibold ${
          isOutline ? "text-teal-dark" : "text-white"
        }`}
      >
        {label}
      </Text>
      {showArrow ? (
        <View className="ml-2">
          <MaterialCommunityIcons
            name="arrow-right"
            size={20}
            color={isOutline ? "#004D4D" : "#ffffff"}
          />
        </View>
      ) : null}
    </Pressable>
  );
}
