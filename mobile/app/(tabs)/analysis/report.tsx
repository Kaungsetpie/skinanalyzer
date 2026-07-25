import { useMemo, useState } from "react";
import { ScrollView, Text, View } from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import type { BackendProduct, FilterChip, ReportTag } from "../../../types/data";
import { AnalysisReportCard } from "../../../components/AnalysisReportCard";
import { FilterBar } from "../../../components/FilterBar";
import { MedicalNote } from "../../../components/MedicalNote";
import { ReportProductCard } from "../../../components/ReportProductCard";
import { skinData } from "../../../lib/skinData";
import { useFetch } from "../../../hooks/requests";

const PRICE_OPTIONS = ["All prices", "Under $10", "$10 - $30", "$30 - $50", "$50+"];
const ALL_PRICE = PRICE_OPTIONS[0]!;
const ALL_ORIGIN = "All regions";

function priceMatches(price: number, filter: string): boolean {
  if (filter === ALL_PRICE) return true;
  if (filter === "Under $10") return price < 10;
  if (filter === "$10 - $30") return price >= 10 && price <= 30;
  if (filter === "$30 - $50") return price > 30 && price <= 50;
  if (filter === "$50+") return price > 50;
  return true;
}

export default function AnalysisReportScreen() {
  const { analysisId } = useLocalSearchParams<{ analysisId: string }>();
  const { data, loading, error } = useFetch(`analysis/${analysisId}`);
  const router = useRouter();

  const [veganOn, setVeganOn] = useState(false);
  const [priceValue, setPriceValue] = useState(ALL_PRICE);
  const [originValue, setOriginValue] = useState(ALL_ORIGIN);

  const fullAnalysis = data?.full_analysis;
  const isSevere: boolean =
    fullAnalysis?.is_severe_requires_clinic ?? data?.dermatologist_required ?? false;

  const allProducts: BackendProduct[] = useMemo(
    () => fullAnalysis?.recommended_products ?? [],
    [fullAnalysis],
  );

  // Build origin picker options from actual product data
  const originOptions = useMemo(() => {
    const countries = [
      ...new Set(allProducts.map((p) => p.country_of_origin).filter(Boolean)),
    ];
    return [ALL_ORIGIN, ...countries];
  }, [allProducts]);

  // Client-side filtering
  const filteredProducts = useMemo(() => {
    return allProducts.filter((p) => {
      const priceOk = priceMatches(p.price, priceValue);
      const originOk =
        originValue === ALL_ORIGIN ||
        p.country_of_origin.toLowerCase().includes(originValue.toLowerCase());
      return priceOk && originOk;
    });
  }, [allProducts, priceValue, originValue]);

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

  const tags: ReportTag[] = useMemo(() => {
    const source: string[] = data?.tags ?? fullAnalysis?.tags ?? [];
    return source.map((t, i) => ({ id: `tag-${i}`, label: t, icon: "tag-outline" }));
  }, [data, fullAnalysis]);

  const { medicalNote } = skinData.report;

  if (loading) {
    return (
      <View className="flex-1 items-center justify-center bg-[#F8F9FA]">
        <Text className="text-slate-600">Analyzing your skin...</Text>
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
            <View className="mt-6">
              <FilterBar
                priceLabel="Price Range"
                priceValue={priceValue}
                priceOptions={PRICE_OPTIONS}
                onPriceChange={setPriceValue}
                originLabel="Made In"
                originValue={originValue}
                originOptions={originOptions}
                onOriginChange={setOriginValue}
                veganLabel="Vegan Choices"
                veganOn={veganOn}
                onVeganChange={setVeganOn}
                matchedCount={filteredProducts.length}
                chips={chips}
                onChipRemove={onChipRemove}
              />
            </View>

            <View className="mt-6">
              {filteredProducts.length === 0 ? (
                <View className="items-center rounded-3xl bg-white py-10">
                  <Text className="text-sm text-slate-500">No products match your filters.</Text>
                </View>
              ) : (
                filteredProducts.map((p, i) => (
                  <ReportProductCard
                    key={`${p.name}-${i}`}
                    product={p}
                    onPress={() =>
                      router.push({
                        pathname: "/(tabs)/analysis/product",
                        params: { product: JSON.stringify(p) },
                      })
                    }
                  />
                ))
              )}
            </View>
          </>
        )}

        <View className="mt-2">
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
