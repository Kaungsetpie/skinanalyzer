import { Text, View } from "react-native";
import { useFetch } from "../../../hooks/requests";
import { AnalysisReportCard } from "../../../components/AnalysisReportCard";

export default function HistoryScreen() {
  const { data, loading, error } = useFetch("analysis/history");

  if (loading) {
    return (
      <View className="flex-1 items-center justify-center bg-[#F8FAFC]">
        <Text className="text-lg font-medium text-slate-900">Loading...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View className="flex-1 items-center justify-center bg-[#F8FAFC]">
        <Text className="text-lg font-medium text-slate-900">
          Error: {String(error)}
        </Text>
      </View>
    );
  }

  if (data && data.length > 0) {
    return (
      <View>
        <View className="flex-1 px-5 pb-28 pt-16">
          <View className="mb-6">
            <Text className="text-2xl font-bold text-slate-900">History</Text>
            <Text className="mt-1 text-sm text-slate-500">
              Your past skin analysis reports.
            </Text>
          </View>
        </View>

        <View>
          {data.map((item: any) => (
            <View className="rounded-3xl bg-slate-100 p-5">
              <Text className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                "Analysis Report"
              </Text>
              <Text className="mt-2 text-xl font-bold leading-7 text-teal-dark">
                {data.headline}
              </Text>
              <Text className="mt-3 text-sm leading-6 text-slate-600">
                {data.summary}
              </Text>

              <View className="mt-4 flex-row flex-wrap gap-2">
                {data.tags.map(
                  (t: { id: string; label: string; icon: string }) => (
                    <View
                      key={t.id}
                      className="flex-row items-center gap-1.5 rounded-full bg-teal-dark px-3 py-1.5"
                    >
                      <Text className="text-xs font-semibold text-white">
                        {t.label}
                      </Text>
                    </View>
                  ),
                )}
              </View>
            </View>
          ))}
        </View>
      </View>
    );
  }

  return (
    <View className="flex-1 items-center justify-center bg-[#F8FAFC]">
      <Text className="text-lg font-medium text-slate-900">
        No analysis history.
      </Text>
    </View>
  );
}
