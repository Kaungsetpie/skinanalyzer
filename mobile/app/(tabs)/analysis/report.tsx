import { useMemo, useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import type { BackendProduct, FilterChip } from "../../../types/data";
import { AnalysisReportCard } from "../../../components/AnalysisReportCard";
import { FilterBar } from "../../../components/FilterBar";
import { MedicalNote } from "../../../components/MedicalNote";
import { ReportProductCard } from "../../../components/ReportProductCard";
import { skinData } from "../../../lib/skinData";
import { useFetch } from "../../../hooks/requests";

const ALL_PRICE = "All prices";
const ALL_ORIGIN = "All regions";

const PRICE_OPTIONS = [ALL_PRICE, "Under $30", "$30 – $80", "$80+"];

function priceMatches(price: number, filter: string): boolean {
  if (filter === ALL_PRICE) return true;
  if (filter === "Under $30") return price < 30;
  if (filter === "$30 – $80") return price >= 30 && price <= 80;
  if (filter === "$80+") return price > 80;
  return true;
}

export default function AnalysisReportScreen() {
  const { analysisId, budget: budgetParam } =
    useLocalSearchParams<{ analysisId: string; budget?: string }>();
  const { data, loading, error } = useFetch(`analysis/${analysisId}`);
  const router = useRouter();

  const userBudget = parseFloat(budgetParam ?? "0");

  const [priceValue, setPriceValue] = useState(ALL_PRICE);
  const [originValue, setOriginValue] = useState(ALL_ORIGIN);

  const fullAnalysis = data?.full_analysis;
  const isSevere: boolean =
    fullAnalysis?.is_severe_requires_clinic ?? data?.dermatologist_required ?? false;

  const allProducts: BackendProduct[] = useMemo(
    () => fullAnalysis?.recommended_products ?? [],
    [fullAnalysis],
  );

  // Split products by user's budget
  const priorityProducts = useMemo(
    () => (userBudget > 0 ? allProducts.filter((p) => p.price <= userBudget) : allProducts),
    [allProducts, userBudget],
  );
  const otherProducts = useMemo(
    () => (userBudget > 0 ? allProducts.filter((p) => p.price > userBudget) : []),
    [allProducts, userBudget],
  );

  // Origin options derived from other products
  const originOptions = useMemo(() => {
    const countries = [
      ...new Set(otherProducts.map((p) => p.country_of_origin).filter(Boolean)),
    ];
    return [ALL_ORIGIN, ...countries];
  }, [otherProducts]);

  // Filtered other options
  const filteredOther = useMemo(
    () =>
      otherProducts.filter((p) => {
        const priceOk = priceMatches(p.price, priceValue);
        const originOk =
          originValue === ALL_ORIGIN ||
          p.country_of_origin.toLowerCase().includes(originValue.toLowerCase());
        return priceOk && originOk;
      }),
    [otherProducts, priceValue, originValue],
  );

  const chips = useMemo(() => {
    const next: FilterChip[] = [];
    if (priceValue !== ALL_PRICE) next.push({ id: "chip-price", label: priceValue });
    if (originValue !== ALL_ORIGIN) next.push({ id: "chip-origin", label: originValue });
    return next;
  }, [priceValue, originValue]);

  const onChipRemove = (id: string) => {
    if (id === "chip-price") setPriceValue(ALL_PRICE);
    if (id === "chip-origin") setOriginValue(ALL_ORIGIN);
  };

  const tags = useMemo(() => {
    const source: string[] = data?.tags ?? fullAnalysis?.tags ?? [];
    return source.map((t: string, i: number) => ({ id: `tag-${i}`, label: t, icon: "tag-outline" }));
  }, [data, fullAnalysis]);

  const { medicalNote } = skinData.report;

  const goToProduct = (p: BackendProduct) =>
    router.push({
      pathname: "/(tabs)/analysis/product",
      params: { product: JSON.stringify(p) },
    });

  if (loading) {
    return (
      <View className="flex-1 items-center justify-center bg-[#F8F9FA]">
        <Text className="text-slate-600">Loading your recommendations…</Text>
      </View>
    );
  }

  if (error || !data) {
    return (
      <View className="flex-1 items-center justify-center bg-[#F8F9FA]">
        <Text className="text-red-500">Failed to load analysis result.</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 bg-[#F8F9FA]">
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-16"
        showsVerticalScrollIndicator={false}
      >
        <AnalysisReportCard
          headline={data.headline ?? "Skin Analysis"}
          summary={data.analysis_summary ?? fullAnalysis?.summary ?? ""}
          scorePercent={isSevere ? 20 : 75}
          scoreLabel={isSevere ? "See a Doctor" : "Good"}
          tags={tags}
        />

        {!isSevere && allProducts.length > 0 && (
          <>
            {/* ── Priority section ── */}
            <View className="mt-8">
              <Text className="mb-1 text-base font-bold text-slate-900">
                Within Your Budget
              </Text>
              <Text className="mb-4 text-xs text-slate-500">
                Products at or below ${userBudget.toFixed(0)} USD, matched to your skin conditions.
              </Text>

              {priorityProducts.length === 0 ? (
                <View className="items-center rounded-3xl bg-white py-8">
                  <Text className="text-sm text-slate-500">
                    No products found within ${userBudget.toFixed(0)}.
                  </Text>
                </View>
              ) : (
                priorityProducts.map((p, i) => (
                  <ReportProductCard
                    key={`priority-${p.name}-${i}`}
                    product={p}
                    onPress={() => goToProduct(p)}
                  />
                ))
              )}
            </View>

            {/* ── Other options section ── */}
            {otherProducts.length > 0 && (
              <View className="mt-8">
                <Text className="mb-1 text-base font-bold text-slate-900">Other Options</Text>
                <Text className="mb-4 text-xs text-slate-500">
                  More products across different price ranges. Filter to explore.
                </Text>

                <FilterBar
                  priceLabel="Price Range"
                  priceValue={priceValue}
                  priceOptions={PRICE_OPTIONS}
                  onPriceChange={setPriceValue}
                  originLabel="Made In"
                  originValue={originValue}
                  originOptions={originOptions}
                  onOriginChange={setOriginValue}
                  veganLabel=""
                  veganOn={false}
                  onVeganChange={() => {}}
                  matchedCount={filteredOther.length}
                  chips={chips}
                  onChipRemove={onChipRemove}
                />

                <View className="mt-4">
                  {filteredOther.length === 0 ? (
                    <View className="items-center rounded-3xl bg-white py-8">
                      <Text className="text-sm text-slate-500">
                        No products match your filters.
                      </Text>
                    </View>
                  ) : (
                    filteredOther.map((p, i) => (
                      <ReportProductCard
                        key={`other-${p.name}-${i}`}
                        product={p}
                        onPress={() => goToProduct(p)}
                      />
                    ))
                  )}
                </View>
              </View>
            )}
          </>
        )}

        <View className="mt-6">
          <MedicalNote
            title={medicalNote.title}
            description={medicalNote.description}
            ctaLabel={medicalNote.ctaLabel}
          />
        </View>
      </ScrollView>
    </View>
  );
}
