import { Stack } from "expo-router";

export default function AnalysisStackLayout() {
  return (
    <Stack screenOptions={{ headerShown: false, animation: "slide_from_right" }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="report" />
      <Stack.Screen name="product" />
    </Stack>
  );
}
