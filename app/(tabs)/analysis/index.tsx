import { useRouter } from "expo-router";
import { ScrollView, Text, View } from "react-native";
import { CaptureCard } from "../../../components/CaptureCard";
import { Header } from "../../../components/Header";
import { InfoCard } from "../../../components/InfoCard";
import { StatusCard } from "../../../components/StatusCard";
import { skinData } from "../../../lib/skinData";

export default function AnalysisScreen() {
  const router = useRouter();
  const { app, user, analysisCapture: a } = skinData;

  const heroPlain = a.heroTitle.replace(a.heroAccent, "").trim();

  return (
    <View className="flex-1 bg-[#F8FAFC]">
      <Header title={app.name} avatarUri={user.avatarUri} />
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-4"
        showsVerticalScrollIndicator={false}
      >
        <View className="mb-6 items-center">
          <Text className="text-center text-2xl font-bold text-slate-900">
            {heroPlain}{" "}
            <Text className="text-teal-primary">{a.heroAccent}</Text>
          </Text>
          <Text className="mt-3 max-w-md text-center text-sm leading-6 text-slate-600">
            {a.heroSubtitle}
          </Text>
        </View>

        <CaptureCard
          liveStatusLabel={a.liveStatusLabel}
          liveStatusDetail={a.liveStatusDetail}
          captureTitle={a.captureTitle}
          captureHint={a.captureHint}
        />

        <View className="mt-5 flex-row gap-3">
          <InfoCard
            icon="white-balance-sunny"
            title={a.optimalLight.title}
            description={a.optimalLight.description}
            variant="teal"
          />
          <InfoCard
            icon="history"
            title={a.lastSession.title}
            description={a.lastSession.description}
            variant="blue"
          />
        </View>

        <View className="mt-5">
          <StatusCard
            title={a.analyzing.title}
            description={a.analyzing.description}
            ctaLabel={a.analyzing.ctaLabel}
            onCtaPress={() => router.push("/(tabs)/analysis/report")}
          />
        </View>
      </ScrollView>
    </View>
  );
}
