import { useState } from "react";
import { useLocalSearchParams, useRouter } from "expo-router";
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { apiRequest } from "../../../services/api";
import conditionsMeta from "../../../data/skin_conditions.json";

type ConditionCard = {
  key: string;
  headline: string;
  description: string;
  tip: string;
  icon: string;
};

function buildConditionCards(conditions: Record<string, any>): ConditionCard[] {
  const cards: ConditionCard[] = [];

  if (conditions.is_combination) {
    const m = conditionsMeta.combination;
    cards.push({ key: "combination", ...m });
  } else if (conditions.skin_type && conditions.skin_type !== "normal") {
    const m = conditionsMeta.skin_type[conditions.skin_type as "oily" | "dry"];
    if (m) cards.push({ key: "skin_type", ...m });
  } else if (conditions.skin_type === "normal") {
    const m = conditionsMeta.skin_type.normal;
    cards.push({ key: "skin_type", ...m });
  }

  if (conditions.acne_type && conditions.acne_type !== "no_acne") {
    const m = conditionsMeta.acne_type[conditions.acne_type as "comedonal_acne" | "inflammatory_acne"];
    if (m) cards.push({ key: "acne_type", ...m });
  }

  if (conditions.has_hyperpigmentation) {
    cards.push({ key: "hyperpigmentation", ...conditionsMeta.hyperpigmentation });
  }

  if (conditions.is_sensitive) {
    cards.push({ key: "sensitive", ...conditionsMeta.sensitive });
  }

  return cards;
}

export default function ConditionsScreen() {
  const router = useRouter();
  const { analysisId, conditions: conditionsParam } =
    useLocalSearchParams<{ analysisId: string; conditions: string }>();

  const conditions: Record<string, any> = JSON.parse(conditionsParam ?? "{}");
  const isSevere = conditions.is_severe === true;

  const [budget, setBudget] = useState("");
  const [loading, setLoading] = useState(false);

  const conditionCards = isSevere ? [] : buildConditionCards(conditions);

  const handleGetRecommendations = async () => {
    const budgetValue = parseFloat(budget);
    if (!budget || isNaN(budgetValue) || budgetValue <= 0) {
      Alert.alert("Enter a budget", "Please enter a valid budget amount in USD.");
      return;
    }

    setLoading(true);
    try {
      await apiRequest(
        `analysis/recommendations/${analysisId}`,
        "POST",
        { budget: budgetValue },
      );
      router.push({
        pathname: "/(tabs)/analysis/report",
        params: { analysisId, budget: String(budgetValue) },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Something went wrong.";
      Alert.alert("Failed", message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View className="flex-1 bg-[#F8FAFC]">
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-32 pt-16"
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View className="mb-6">
          <Text className="text-2xl font-bold text-slate-900">Your Skin Analysis</Text>
          <Text className="mt-2 text-sm leading-6 text-slate-500">
            {isSevere
              ? "A potentially serious condition was detected."
              : "Here's what we found. Review your conditions below, then enter your budget for personalised product recommendations."}
          </Text>
        </View>

        {/* Severe warning */}
        {isSevere && (
          <View className="mb-4 rounded-3xl bg-red-50 p-5">
            <View className="mb-3 flex-row items-center gap-3">
              <View className="h-10 w-10 items-center justify-center rounded-full bg-red-100">
                <MaterialCommunityIcons name="hospital-box-outline" size={22} color="#dc2626" />
              </View>
              <Text className="text-base font-bold text-red-700">
                {conditionsMeta.severe.headline}
              </Text>
            </View>
            <Text className="text-sm leading-6 text-red-600">
              {conditionsMeta.severe.description}
            </Text>
            <Text className="mt-2 text-xs font-medium text-red-500">
              {conditionsMeta.severe.tip}
            </Text>
          </View>
        )}

        {/* Condition cards */}
        {conditionCards.map((card) => (
          <View key={card.key} className="mb-4 rounded-3xl bg-white p-5 shadow-sm">
            <View className="mb-3 flex-row items-center gap-3">
              <View className="h-10 w-10 items-center justify-center rounded-full bg-teal-50">
                <MaterialCommunityIcons
                  name={card.icon as any}
                  size={20}
                  color="#00797C"
                />
              </View>
              <Text className="text-base font-bold text-slate-900">{card.headline}</Text>
            </View>
            <Text className="text-sm leading-6 text-slate-600">{card.description}</Text>
            <View className="mt-3 rounded-xl bg-teal-50 px-4 py-3">
              <Text className="text-xs font-medium leading-5 text-teal-700">{card.tip}</Text>
            </View>
          </View>
        ))}

        {/* Budget input — only for non-severe */}
        {!isSevere && (
          <View className="mt-2 rounded-3xl bg-white p-5 shadow-sm">
            <Text className="mb-1 text-xs font-bold uppercase tracking-wider text-slate-400">
              Your Budget (USD)
            </Text>
            <Text className="mb-3 text-xs text-slate-500">
              We'll prioritise products within your budget, plus show you other options.
            </Text>
            <TextInput
              value={budget}
              onChangeText={setBudget}
              keyboardType="numeric"
              placeholder="e.g. 50"
              placeholderTextColor="#94a3b8"
              className="rounded-xl border border-slate-200 px-4 py-3 text-base text-slate-900"
            />
          </View>
        )}
      </ScrollView>

      {/* CTA */}
      {!isSevere && (
        <View className="absolute bottom-0 left-0 right-0 border-t border-slate-100 bg-white px-5 pb-10 pt-4">
          <Pressable
            onPress={handleGetRecommendations}
            disabled={loading}
            className="items-center rounded-2xl bg-teal-600 py-4"
            style={({ pressed }) => ({ opacity: pressed || loading ? 0.7 : 1 })}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text className="text-base font-bold text-white">Get Recommendations</Text>
            )}
          </Pressable>
        </View>
      )}

      {isSevere && (
        <View className="absolute bottom-0 left-0 right-0 border-t border-slate-100 bg-white px-5 pb-10 pt-4">
          <Pressable
            onPress={() => router.back()}
            className="items-center rounded-2xl bg-slate-800 py-4"
            style={({ pressed }) => ({ opacity: pressed ? 0.7 : 1 })}
          >
            <Text className="text-base font-bold text-white">Go Back</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}
