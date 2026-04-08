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

          <View className="mt-8 gap-4">
            {home.features.map((f) => (
              <FeatureCard
                key={f.id}
                feature={f}
                onDemoPress={() => router.push("/(tabs)/analysis")}
              />
            ))}
          </View>
          <View className="mt-8 overflow-hidden rounded-3xl">
            <LinearGradient
              colors={["#006D77", "#004D4D"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={{ padding: 24 }}
            >
              <Text className="text-xl font-bold text-white">
                {home.footerCard.title}
              </Text>
              <Text className="mt-2 text-sm leading-6 text-white/85">
                {home.footerCard.subtitle}
              </Text>
              <View className="mt-4 h-28 items-center justify-center rounded-2xl bg-white/10">
                <MaterialCommunityIcons name="flask" size={48} color="#ffffff" />
              </View>
            </LinearGradient>
          </View>
        </ScrollView>

        <Pressable
          className="absolute bottom-24 right-6 h-12 w-12 items-center justify-center rounded-2xl bg-teal-alt shadow-lg active:opacity-90"
          onPress={() => {}}
          accessibilityRole="button"
          accessibilityLabel="Settings"
        >
          <MaterialCommunityIcons name="cog-outline" size={26} color="#ffffff" />
        </Pressable>
      </View>
    </View>
  );
}
