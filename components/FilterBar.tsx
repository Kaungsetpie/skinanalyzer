import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Pressable, Switch, Text, View } from "react-native";
import type { FilterChip } from "../types/data";

type FilterBarProps = {
  priceLabel: string;
  priceValue: string;
  originLabel: string;
  originValue: string;
  veganLabel: string;
  veganOn: boolean;
  onVeganChange?: (v: boolean) => void;
  matchedCount: number;
  chips: FilterChip[];
  onChipRemove?: (id: string) => void;
};

export function FilterBar({
  priceLabel,
  priceValue,
  originLabel,
  originValue,
  veganLabel,
  veganOn,
  onVeganChange,
  matchedCount,
  chips,
  onChipRemove,
}: FilterBarProps) {
  return (
    <View className="gap-3">
      <View className="flex-row gap-3">
        <Pressable className="flex-1 flex-row items-center justify-between rounded-2xl border border-slate-200 bg-white px-3 py-3 active:bg-slate-50">
          <Text className="text-xs font-semibold text-slate-700">{priceLabel}</Text>
          <View className="flex-row items-center gap-1">
            <Text className="text-xs text-slate-500">{priceValue}</Text>
            <MaterialCommunityIcons name="chevron-down" size={18} color="#64748b" />
          </View>
        </Pressable>
        <Pressable className="flex-1 flex-row items-center justify-between rounded-2xl border border-slate-200 bg-white px-3 py-3 active:bg-slate-50">
          <Text className="text-xs font-semibold text-slate-700">{originLabel}</Text>
          <View className="flex-row items-center gap-1">
            <Text className="text-xs text-slate-500">{originValue}</Text>
            <MaterialCommunityIcons name="chevron-down" size={18} color="#64748b" />
          </View>
        </Pressable>
      </View>

      <View className="flex-row items-center justify-between rounded-2xl bg-white px-3 py-2">
        <Text className="text-sm font-medium text-slate-800">{veganLabel}</Text>
        <Switch
          value={veganOn}
          onValueChange={onVeganChange}
          trackColor={{ false: "#cbd5e1", true: "#99f6e4" }}
          thumbColor={veganOn ? "#00797C" : "#f4f4f5"}
        />
      </View>

      <Text className="text-xs font-semibold text-slate-500">
        {matchedCount} Results Matched
      </Text>

      <View className="flex-row flex-wrap gap-2">
        {chips.map((chip) => (
          <Pressable
            key={chip.id}
            onPress={() => onChipRemove?.(chip.id)}
            className="flex-row items-center gap-1 rounded-full bg-[#D1E9FF] px-3 py-1.5 active:opacity-80"
          >
            <Text className="text-xs font-medium text-slate-800">{chip.label}</Text>
            <MaterialCommunityIcons name="close" size={14} color="#0f172a" />
          </Pressable>
        ))}
      </View>
    </View>
  );
}
