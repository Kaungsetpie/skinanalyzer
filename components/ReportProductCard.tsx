import { Image, Text, View } from "react-native";
import type { ReportProduct } from "../types/data";
import { ActionButton } from "./ActionButton";

type ReportProductCardProps = {
  product: ReportProduct;
  onAddPress?: () => void;
};

export function ReportProductCard({ product, onAddPress }: ReportProductCardProps) {
  const metaLeft = [product.origin, ...product.badges].join(" • ");

  return (
    <View className="mb-4 overflow-hidden rounded-3xl bg-white p-4 shadow-sm">
      <View className="flex-row gap-4">
        <View className="h-24 w-24 overflow-hidden rounded-2xl bg-slate-100">
          <Image
            source={{ uri: product.imageUri }}
            className="h-full w-full"
            resizeMode="cover"
          />
        </View>
        <View className="flex-1">
          <View className="flex-row items-start justify-between gap-2">
            <Text className="flex-1 text-[10px] font-bold uppercase tracking-wide text-slate-500">
              {metaLeft}
            </Text>
            <Text className="text-sm font-bold text-slate-900">{product.price}</Text>
          </View>
          <Text className="mt-1 text-base font-bold text-slate-900">{product.name}</Text>
          <Text
            className="mt-1 text-xs leading-5 text-slate-500"
            numberOfLines={2}
          >
            {product.description}
          </Text>
        </View>
      </View>
      <View className="mt-4">
        <ActionButton
          label={product.ctaLabel}
          onPress={onAddPress}
          colorClass="bg-teal-dark"
        />
      </View>
    </View>
  );
}
