import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Text, View } from "react-native";
import { ActionButton } from "./ActionButton";

type MedicalNoteProps = {
  title: string;
  description: string;
  ctaLabel: string;
  onCtaPress?: () => void;
};

export function MedicalNote({
  title,
  description,
  ctaLabel,
  onCtaPress,
}: MedicalNoteProps) {
  return (
    <View className="rounded-3xl bg-slate-200/80 p-5">
      <View className="flex-row gap-4">
        <View className="h-12 w-12 items-center justify-center rounded-xl bg-teal-dark">
          <MaterialCommunityIcons name="medical-bag" size={26} color="#fff" />
        </View>
        <View className="flex-1">
          <Text className="text-lg font-bold text-slate-900">{title}</Text>
          <Text className="mt-2 text-sm leading-6 text-slate-600">{description}</Text>
        </View>
      </View>
      <View className="mt-4">
        <ActionButton
          label={ctaLabel}
          onPress={onCtaPress}
          variant="outline"
          colorClass="bg-white"
        />
      </View>
    </View>
  );
}
