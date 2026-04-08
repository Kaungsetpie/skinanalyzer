import { useRouter } from "expo-router";
import { useState } from "react";
import { ScrollView, View } from "react-native";
import { AnalysisReportCard } from "../../../components/AnalysisReportCard";
import { FilterBar } from "../../../components/FilterBar";
import { Header } from "../../../components/Header";
import { MedicalNote } from "../../../components/MedicalNote";
import { ReportProductCard } from "../../../components/ReportProductCard";
import { skinData } from "../../../lib/skinData";

export default function AnalysisReportScreen() {
  const router = useRouter();
  const { app, user, report } = skinData;
  const [veganOn, setVeganOn] = useState(report.filters.veganOn);

  return (
    <View className="flex-1 bg-[#F8F9FA]">
      <Header
        title={app.name}
        avatarUri={user.avatarUri}
        variant="back"
        onBackPress={() => router.back()}
      />
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-4"
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
            priceLabel={report.filters.priceLabel}
            priceValue={report.filters.priceValue}
            originLabel={report.filters.originLabel}
            originValue={report.filters.originValue}
            veganLabel={report.filters.veganLabel}
            veganOn={veganOn}
            onVeganChange={setVeganOn}
            matchedCount={report.filters.matchedCount}
            chips={report.filters.activeChips}
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
