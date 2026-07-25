import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useCallback, useMemo, useState } from "react";
import {
  Modal,
  Pressable,
  ScrollView,
  Switch,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import type { FilterChip } from "../../types/data";

type PickerKind = "price" | "origin" | null;

type FilterBarProps = {
  priceLabel: string;
  priceValue: string;
  priceOptions: string[];
  onPriceChange?: (value: string) => void;
  originLabel: string;
  originValue: string;
  originOptions: string[];
  onOriginChange?: (value: string) => void;
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
  priceOptions,
  onPriceChange,
  originLabel,
  originValue,
  originOptions,
  onOriginChange,
  veganLabel,
  veganOn,
  onVeganChange,
  matchedCount,
  chips,
  onChipRemove,
}: FilterBarProps) {
  const insets = useSafeAreaInsets();
  const [picker, setPicker] = useState<PickerKind>(null);

  const pickerTitle = useMemo(() => {
    if (picker === "price") return priceLabel;
    if (picker === "origin") return originLabel;
    return "";
  }, [picker, priceLabel, originLabel]);

  const pickerOptions = picker === "price" ? priceOptions : originOptions;
  const selectedValue = picker === "price" ? priceValue : originValue;
  const onSelect =
    picker === "price" ? onPriceChange : picker === "origin" ? onOriginChange : undefined;

  const closePicker = useCallback(() => setPicker(null), []);

  const handleSelect = useCallback(
    (value: string) => {
      onSelect?.(value);
      setPicker(null);
    },
    [onSelect],
  );

  return (
    <View className="gap-3">
      <View className="flex-row gap-3">
        <Pressable
          onPress={() => setPicker("price")}
          className="flex-1 flex-row items-center justify-between rounded-2xl border border-slate-200 bg-white px-3 py-3 active:bg-slate-50"
          accessibilityRole="button"
          accessibilityLabel={`${priceLabel}, current ${priceValue}`}
        >
          <Text className="text-xs font-semibold text-slate-700">{priceLabel}</Text>
          <View className="flex-row items-center gap-1">
            <Text className="text-xs text-slate-500">{priceValue}</Text>
            <MaterialCommunityIcons name="chevron-down" size={18} color="#64748b" />
          </View>
        </Pressable>
        <Pressable
          onPress={() => setPicker("origin")}
          className="flex-1 flex-row items-center justify-between rounded-2xl border border-slate-200 bg-white px-3 py-3 active:bg-slate-50"
          accessibilityRole="button"
          accessibilityLabel={`${originLabel}, current ${originValue}`}
        >
          <Text className="text-xs font-semibold text-slate-700">{originLabel}</Text>
          <View className="flex-row items-center gap-1">
            <Text className="text-xs text-slate-500">{originValue}</Text>
            <MaterialCommunityIcons name="chevron-down" size={18} color="#64748b" />
          </View>
        </Pressable>
      </View>

      <Modal
        visible={picker !== null}
        transparent
        animationType="fade"
        onRequestClose={closePicker}
      >
        <View className="flex-1 justify-end">
          <Pressable
            className="absolute bottom-0 left-0 right-0 top-0 bg-black/45"
            onPress={closePicker}
            accessibilityRole="button"
            accessibilityLabel="Dismiss"
          />
          <View
            className="z-10 rounded-t-3xl bg-white shadow-2xl"
            style={{ paddingBottom: Math.max(insets.bottom, 16) }}
          >
            <View className="flex-row items-center justify-between border-b border-slate-100 px-4 py-3">
              <Text className="text-lg font-bold text-slate-900">{pickerTitle}</Text>
              <Pressable
                onPress={closePicker}
                className="h-9 w-9 items-center justify-center rounded-full active:bg-slate-100"
                accessibilityRole="button"
                accessibilityLabel="Close"
              >
                <MaterialCommunityIcons name="close" size={22} color="#64748b" />
              </Pressable>
            </View>
            <ScrollView
              className="max-h-80"
              keyboardShouldPersistTaps="handled"
              showsVerticalScrollIndicator={false}
            >
              {pickerOptions.map((item, index) => {
                const isSelected = item === selectedValue;
                return (
                  <Pressable
                    key={`${item}-${index}`}
                    onPress={() => handleSelect(item)}
                    className={`mx-3 my-1 flex-row items-center justify-between rounded-2xl px-4 py-3.5 active:bg-slate-50 ${
                      isSelected ? "bg-[#E0F7F8]" : "bg-white"
                    }`}
                    accessibilityRole="button"
                    accessibilityState={{ selected: isSelected }}
                  >
                    <Text
                      className={`text-base ${
                        isSelected ? "font-semibold text-teal-dark" : "font-medium text-slate-800"
                      }`}
                    >
                      {item}
                    </Text>
                    {isSelected ? (
                      <MaterialCommunityIcons name="check" size={22} color="#004D4D" />
                    ) : null}
                  </Pressable>
                );
              })}
            </ScrollView>
          </View>
        </View>
      </Modal>

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
