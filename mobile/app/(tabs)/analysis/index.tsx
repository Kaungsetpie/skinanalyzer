import { useState } from "react";
import { useRouter } from "expo-router";
import { ScrollView, Text, View, Image } from "react-native";
import { CaptureCard } from "../../../components/CaptureCard";
import { InfoCard } from "../../../components/InfoCard";
import { StatusCard } from "../../../components/StatusCard";
import { skinData } from "../../../lib/skinData";
import * as ImagePicker from "expo-image-picker";
import { useFetch } from "../../../hooks/useFetch";
import { apiRequest } from "../../../services/api";

export default function AnalysisScreen() {
  const router = useRouter();
  const { app, user, analysisCapture: a } = skinData;
  const [image, setImage] = useState<string | null>(null);
  const heroPlain = a.heroTitle.replace(a.heroAccent, "").trim();
  const sendImageToServer = async (imageFile: string) => {
    const formData = new FormData();
    formData.append('file', {
      uri: imageFile,
      name: 'photo.jpg',
      type: 'image/jpeg'
    } as any);
   const result = await apiRequest("analysis/upload", "POST", formData,"formData");
   console.log("Upload result:", result);
  }
  const uploadPhoto = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });
    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
    if (result && result.assets) {
      await sendImageToServer(result.assets[0]?.uri);
    }
  }
  const takePhoto = async () => {
    let result = await ImagePicker.requestCameraPermissionsAsync();
    if (result.granted) {
      let res = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });
      if (!res.canceled) {
        setImage(res.assets[0].uri);
      }
      if (res && res.assets) {
        await sendImageToServer(res.assets[0]?.uri);
      }
    }
  }
  return (
    <View className="flex-1 bg-[#F8FAFC]">
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-16"
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
          onGalleryPress={uploadPhoto}
          onShutterPress={takePhoto}
        />
        {/*}
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
        */}
        {image && (
          <View className="mt-5">
            <Image source={{ uri: image }} className="h-64 w-full rounded-2xl object-cover" />
            <StatusCard
              title={a.analyzing.title}
              description={a.analyzing.description}
              ctaLabel={a.analyzing.ctaLabel}
              onCtaPress={() => router.push("/(tabs)/analysis/report")}
            />
          </View>
        )
        }
      </ScrollView>
    </View>
  );
}
