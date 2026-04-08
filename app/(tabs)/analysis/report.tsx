import { useMemo, useState } from "react";
import { ScrollView, View } from "react-native";
import type { FilterChip } from "../../../types/data";
import { AnalysisReportCard } from "../../../components/AnalysisReportCard";
import { FilterBar } from "../../../components/FilterBar";
import { MedicalNote } from "../../../components/MedicalNote";
import { ReportProductCard } from "../../../components/ReportProductCard";
import { skinData } from "../../../lib/skinData";

export default function AnalysisReportScreen() {
  const { report } = skinData;
  const f = report.filters;
  const allPrice = f.priceOptions[0]!;
  const allOrigin = f.originOptions[0]!;

  const [veganOn, setVeganOn] = useState(f.veganOn);
  const [priceValue, setPriceValue] = useState(f.priceValue);
  const [originValue, setOriginValue] = useState(f.originValue);

  const chips = useMemo(() => {
    const next: FilterChip[] = [];
    if (priceValue !== allPrice) {
      next.push({ id: "chip-price", label: priceValue });
    }
    if (originValue !== allOrigin) {
      next.push({ id: "chip-origin", label: originValue });
    }
    return next;
  }, [priceValue, originValue, allPrice, allOrigin]);

  const onChipRemove = (id: string) => {
    if (id === "chip-price") setPriceValue(allPrice);
    if (id === "chip-origin") setOriginValue(allOrigin);
  };

  return (
    <View className="flex-1 bg-[#F8F9FA]">
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-16"
        showsVerticalScrollIndicator={false}
      >
        <AnalysisReportCard
          label={report.analysis.label}
          headline={report.analysis.headline}
          summary={report.analysis.summary}
          scorePercent={report.analysis.scorePercent}
          scoreLabel={report.analysis.scoreLabel}
          tags={report.analysis.tags}
        />

        <View className="mt-6">
          <FilterBar
            priceLabel={f.priceLabel}
            priceValue={priceValue}
            priceOptions={f.priceOptions}
            onPriceChange={setPriceValue}
            originLabel={f.originLabel}
            originValue={originValue}
            originOptions={f.originOptions}
            onOriginChange={setOriginValue}
            veganLabel={f.veganLabel}
            veganOn={veganOn}
            onVeganChange={setVeganOn}
            matchedCount={f.matchedCount}
            chips={chips}
            onChipRemove={onChipRemove}
          />
        </View>

        <View className="mt-6">
          {report.products.map((p) => (
            <ReportProductCard key={p.id} product={p} />
          ))}
        </View>

        <View className="mt-2">
          <MedicalNote
            title={report.medicalNote.title}
            description={report.medicalNote.description}
            ctaLabel={report.medicalNote.ctaLabel}
          />
        </View>
      </ScrollView>
    </View>
  );
}
