import { useState } from "react";
import { useRouter } from "expo-router";
import { ActivityIndicator, Alert, Image, ScrollView, Text, View } from "react-native";
import { CaptureCard } from "../../../components/CaptureCard";
import { skinData } from "../../../lib/skinData";
import * as ImagePicker from "expo-image-picker";
import { apiRequest } from "../../../services/api";

export default function AnalysisScreen() {
  const router = useRouter();
  const { analysisCapture: a } = skinData;
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const heroPlain = a.heroTitle.replace(a.heroAccent, "").trim();

  const sendImageToServer = async (imageFile: string) => {
    const formData = new FormData();
    formData.append("file", {
      uri: imageFile,
      name: "photo.jpg",
      type: "image/jpeg",
    } as any);

    setAnalyzing(true);
    try {
      const result = await apiRequest("analysis/upload", "POST", formData, "formData");
      if (result?.analysisId) {
        router.push({
          pathname: "/(tabs)/analysis/conditions",
          params: {
            analysisId: result.analysisId,
            conditions: JSON.stringify(result.conditions),
          },
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Something went wrong.";
      Alert.alert("Analysis Failed", message);
    } finally {
      setAnalyzing(false);
    }
  };

  const uploadPhoto = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });
    if (!result.canceled && result.assets) {
      setImage(result.assets[0].uri);
      await sendImageToServer(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) return;
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });
    if (!result.canceled && result.assets) {
      setImage(result.assets[0].uri);
      await sendImageToServer(result.assets[0].uri);
    }
  };

  if (analyzing) {
    return (
      <View className="flex-1 items-center justify-center bg-[#F8FAFC] px-8">
        {image && (
          <Image
            source={{ uri: image }}
            className="mb-8 h-44 w-44 rounded-3xl"
            resizeMode="cover"
          />
        )}
        <ActivityIndicator size="large" color="#00797C" />
        <Text className="mt-5 text-center text-lg font-bold text-slate-900">
          {a.analyzing.title}
        </Text>
        <Text className="mt-2 text-center text-sm leading-6 text-slate-500">
          {a.analyzing.description}
        </Text>
      </View>
    );
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
      </ScrollView>
    </View>
  );
}
