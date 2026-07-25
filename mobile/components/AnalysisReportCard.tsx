import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Text, View } from "react-native";
import type { ReportTag } from "../../types/data";
import { DonutScore } from "./DonutScore";

function tagIcon(name: string): keyof typeof MaterialCommunityIcons.glyphMap {
  const map: Record<string, keyof typeof MaterialCommunityIcons.glyphMap> = {
    "grid-large": "grid-large",
    "alert-circle-outline": "alert-circle-outline",
  };
  return map[name] ?? "tag-outline";
}

type AnalysisReportCardProps = {
  headline: string;
  summary: string;
  scorePercent: number;
  scoreLabel: string;
  tags: ReportTag[];
};

export function AnalysisReportCard({
  headline,
  summary,
  scorePercent,
  scoreLabel,
  tags,
}: AnalysisReportCardProps) {
  return (
    <View className="rounded-3xl bg-slate-100 p-5">
      <Text className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        "Analysis Report"
      </Text>
      <Text className="mt-2 text-xl font-bold leading-7 text-teal-dark">
        {headline}
      </Text>
      <Text className="mt-3 text-sm leading-6 text-slate-600">{summary}</Text>

      <View className="mt-4 flex-row flex-wrap gap-2">
        {tags.map((t) => (
          <View
            key={t.id}
            className="flex-row items-center gap-1.5 rounded-full bg-teal-dark px-3 py-1.5"
          >
            <MaterialCommunityIcons
              name={tagIcon(t.icon)}
              size={16}
              color="#fff"
            />
            <Text className="text-xs font-semibold text-white">{t.label}</Text>
          </View>
        ))}
      </View>

      <View className="mt-6 items-center">
        <DonutScore percent={scorePercent} label={scoreLabel} />
      </View>
    </View>
  );
}
