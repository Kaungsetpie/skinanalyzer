import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Pressable, ScrollView, Text, View } from "react-native";
import type { BackendProduct } from "../../../types/data";

const BRAND_BG = [
  "#ccfbf1", "#ede9fe", "#ffe4e6",
  "#fef3c7", "#e0f2fe", "#d1fae5",
];
const BRAND_FG = [
  "#0f766e", "#6d28d9", "#be123c",
  "#b45309", "#0369a1", "#065f46",
];

function colorIndex(name: string) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) & 0xff;
  return h % BRAND_BG.length;
}

export default function ProductDetailScreen() {
  const router = useRouter();
  const { product: productParam } =
    useLocalSearchParams<{ product: string }>();

  const product: BackendProduct = JSON.parse(productParam ?? "{}");
  const ci = colorIndex(product.brand ?? "");
  const bg = BRAND_BG[ci]!;
  const fg = BRAND_FG[ci]!;
  const initial = (product.brand ?? "?").charAt(0).toUpperCase();

  return (
    <View className="flex-1 bg-[#F8F9FA]">
      {/* Coloured header band */}
      <View style={{ backgroundColor: bg }} className="pb-6 pt-14">
        <Pressable
          onPress={() => router.back()}
          style={{ backgroundColor: "rgba(255,255,255,0.55)" }}
          className="mx-5 mb-4 h-9 w-9 items-center justify-center rounded-full"
          accessibilityRole="button"
          accessibilityLabel="Go back"
        >
          <MaterialCommunityIcons name="arrow-left" size={20} color="#0f172a" />
        </Pressable>

        <View className="mx-5 flex-row items-center gap-4">
          <View
            style={{ backgroundColor: "rgba(255,255,255,0.55)" }}
            className="h-20 w-20 items-center justify-center rounded-2xl"
          >
            <Text style={{ color: fg }} className="text-4xl font-bold">
              {initial}
            </Text>
          </View>
          <View className="flex-1">
            <Text className="text-xs font-semibold uppercase tracking-widest text-slate-500">
              {product.brand}
            </Text>
            <Text className="mt-0.5 text-xl font-bold leading-7 text-slate-900">
              {product.name}
            </Text>
            <Text className="mt-1 text-sm font-medium text-slate-500">
              {product.country_of_origin}
              {product.price != null ? `  ·  $${product.price.toFixed(2)}` : ""}
            </Text>
          </View>
        </View>
      </View>

      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-6"
        showsVerticalScrollIndicator={false}
      >
        {/* Description */}
        <View className="rounded-3xl bg-white p-5 shadow-sm">
          <Text className="text-xs font-bold uppercase tracking-wider text-slate-400">
            About this product
          </Text>
          <Text className="mt-2 text-sm leading-6 text-slate-700">
            {product.description}
          </Text>
        </View>

        {/* How it helps your skin — Gemini-generated, personalised to detected conditions */}
        {!!product.benefits && (
          <View className="mt-4 rounded-3xl bg-white p-5 shadow-sm">
            <View className="flex-row items-center gap-2">
              <View
                style={{ backgroundColor: bg }}
                className="h-7 w-7 items-center justify-center rounded-full"
              >
                <MaterialCommunityIcons name="heart-pulse" size={14} color={fg} />
              </View>
              <Text className="text-xs font-bold uppercase tracking-wider text-slate-400">
                How it helps your skin
              </Text>
            </View>
            <Text className="mt-3 text-sm leading-6 text-slate-700">
              {product.benefits}
            </Text>
          </View>
        )}

        {/* Key ingredients */}
        {product.key_ingredients?.length > 0 && (
          <View className="mt-4 rounded-3xl bg-white p-5 shadow-sm">
            <Text className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Key ingredients
            </Text>
            <View className="mt-3 flex-row flex-wrap gap-2">
              {product.key_ingredients.map((ing: string) => (
                <View
                  key={ing}
                  style={{ backgroundColor: bg }}
                  className="rounded-full px-3 py-1.5"
                >
                  <Text style={{ color: fg }} className="text-xs font-semibold">
                    {ing}
                  </Text>
                </View>
              ))}
            </View>
          </View>
        )}
      </ScrollView>
    </View>
  );
}
