import { MaterialCommunityIcons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { Pressable, ScrollView, Text, View } from "react-native";
import { ActionButton } from "../../components/ActionButton";
import { FeatureCard } from "../../components/FeatureCard";
import { Header } from "../../components/Header";
import { HomeProductItem } from "../../components/HomeProductItem";
import { LiveAnalysisHero } from "../../components/LiveAnalysisHero";
import { skinData } from "../../lib/skinData";

function Headline() {
  const { headline, headlineAccent } = skinData.home;
  const i = headline.indexOf(headlineAccent);
  if (i < 0) {
    return (
      <Text className="text-center text-3xl font-bold leading-tight text-slate-900">
        {headline}
      </Text>
    );
  }
  const before = headline.slice(0, i);
  const after = headline.slice(i + headlineAccent.length);
  return (
    <Text className="text-center text-3xl font-bold leading-tight text-slate-900">
      {before}
      <Text className="font-bold italic text-teal-alt">{headlineAccent}</Text>
      {after}
    </Text>
  );
}

export default function HomeScreen() {
  const router = useRouter();
  const { app, user, home } = skinData;

  return (
    <View className="flex-1 bg-[#F8F9FA]">
      {/*  <Header title={app.name} avatarUri={user.avatarUri} /> */}

      <View className="flex-1">
        <ScrollView
          className="flex-1"
          contentContainerClassName="px-6 pb-36 pt-16"
          showsVerticalScrollIndicator={false}
        >
          <View className="mb-4 self-center rounded-full bg-[#E0F2F1] px-4 py-1.5">
            <Text className="text-[10px] font-bold tracking-widest text-teal-alt">
              {home.badge}
            </Text>
          </View>

          <Headline />

          <Text className="mt-4 text-center text-sm leading-6 text-slate-600">
            {home.subtext}
          </Text>

          <View className="mt-6 gap-3">
            <ActionButton
              label={home.primaryCta}
              showArrow
              colorClass="bg-teal-alt"
              onPress={() => router.push("/(tabs)/analysis")}
            />
          </View>

          <View className="mt-8">
            <LiveAnalysisHero
              imageUri={home.liveAnalysis.imageUri}
              tag={home.liveAnalysis.tag}
              title={home.liveAnalysis.title}
              scanningLabel={home.liveAnalysis.scanningLabel}
              hydrationLabel={home.liveAnalysis.hydrationLabel}
              hydrationPercent={home.liveAnalysis.hydrationPercent}
            />
          </View>
        </ScrollView>
      </View>
    </View>
  );
}
