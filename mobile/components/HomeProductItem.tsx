import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Image, Pressable, Text, View } from "react-native";
import type { HomeProduct } from "../../types/data";

type HomeProductItemProps = {
  product: HomeProduct;
  onShopPress?: () => void;
};

export function HomeProductItem({ product, onShopPress }: HomeProductItemProps) {
  return (
    <View className="mr-4 w-64 overflow-hidden rounded-3xl bg-white shadow-sm">
      <View className="h-40 bg-slate-100">
        <Image
          source={{ uri: product.imageUri }}
          className="h-full w-full"
          resizeMode="cover"
        />
      </View>
      <View className="p-4">
        <Text className="text-base font-bold text-slate-900">{product.name}</Text>
        <Text className="mt-1 text-xs leading-5 text-slate-500">
          {product.recommendationText}
        </Text>
        <Pressable
          onPress={onShopPress}
          className="mt-3 flex-row items-center gap-1 active:opacity-70"
        >
          <Text className="text-sm font-semibold text-teal-alt">
            {product.shopLabel}
          </Text>
          <MaterialCommunityIcons name="chevron-right" size={18} color="#006D77" />
        </Pressable>
      </View>
    </View>
  );
}
