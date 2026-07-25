import { Pressable, Text, View } from "react-native";
import type { BackendProduct } from "../types/data";

const BRAND_COLORS = [
  "bg-teal-100", "bg-violet-100", "bg-rose-100",
  "bg-amber-100", "bg-sky-100", "bg-emerald-100",
];

const BRAND_TEXT_COLORS = [
  "text-teal-700", "text-violet-700", "text-rose-700",
  "text-amber-700", "text-sky-700", "text-emerald-700",
];

function brandColorIndex(name: string) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xff;
  return h % BRAND_COLORS.length;
}

type ReportProductCardProps = {
  product: BackendProduct;
  onPress?: () => void;
};

export function ReportProductCard({ product, onPress }: ReportProductCardProps) {
  const ci = brandColorIndex(product.brand);
  const initial = product.brand.charAt(0).toUpperCase();
  const formattedPrice = `$${product.price.toFixed(2)}`;
  const meta = [product.brand, product.country_of_origin].filter(Boolean).join(" • ").toUpperCase();

  return (
    <Pressable
      onPress={onPress}
      className="mb-4 overflow-hidden rounded-3xl bg-white p-4 shadow-sm active:opacity-75"
    >
      <View className="flex-row gap-4">
        <View
          className={`h-24 w-24 shrink-0 items-center justify-center rounded-2xl ${BRAND_COLORS[ci]}`}
        >
          <Text className={`text-3xl font-bold ${BRAND_TEXT_COLORS[ci]}`}>{initial}</Text>
        </View>

        <View className="flex-1">
          <View className="flex-row items-start justify-between gap-2">
            <Text className="flex-1 text-[10px] font-bold uppercase tracking-wide text-slate-500">
              {meta}
            </Text>
            <Text className="text-sm font-bold text-slate-900">{formattedPrice}</Text>
          </View>

          <Text className="mt-1 text-base font-bold text-slate-900">{product.name}</Text>

          <Text className="mt-1 text-xs leading-5 text-slate-500" numberOfLines={2}>
            {product.description}
          </Text>
        </View>
      </View>

      {product.key_ingredients.length > 0 && (
        <View className="mt-3 flex-row flex-wrap gap-1.5">
          {product.key_ingredients.slice(0, 4).map((ing: string) => (
            <View key={ing} className="rounded-full bg-slate-100 px-2.5 py-1">
              <Text className="text-[10px] font-semibold text-slate-600">{ing}</Text>
            </View>
          ))}
        </View>
      )}
    </Pressable>
  );
}
