import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Image, Pressable, ScrollView, Text, View } from "react-native";
import { Header } from "../../components/Header";
import { skinData } from "../../lib/skinData";

function rowIcon(
  name: string,
): keyof typeof MaterialCommunityIcons.glyphMap {
  const map: Record<string, keyof typeof MaterialCommunityIcons.glyphMap> = {
    history: "history",
    "bell-outline": "bell-outline",
    "lock-outline": "lock-outline",
  };
  return map[name] ?? "chevron-right";
}

export default function ProfileScreen() {
  const { app, user, profile } = skinData;

  return (
    <View className="flex-1 bg-[#F8FAFC]">
      <Header title={app.name} avatarUri={user.avatarUri} />
      <ScrollView
        className="flex-1"
        contentContainerClassName="px-5 pb-28 pt-6"
        showsVerticalScrollIndicator={false}
      >
        <View className="items-center">
          <View className="h-24 w-24 overflow-hidden rounded-full border-4 border-white shadow-md">
            <Image
              source={{ uri: user.avatarUri }}
              className="h-full w-full"
              resizeMode="cover"
            />
          </View>
          <Text className="mt-4 text-2xl font-bold text-slate-900">
            {user.displayName}
          </Text>
          <Text className="mt-1 text-center text-sm text-slate-500">
            {profile.subtext}
          </Text>
        </View>

        <Text className="mt-8 text-lg font-bold text-slate-900">
          {profile.headline}
        </Text>

        <View className="mt-4 overflow-hidden rounded-3xl bg-white shadow-sm">
          {profile.rows.map((row, index) => (
            <Pressable
              key={row.id}
              onPress={() => {}}
              className={`flex-row items-center justify-between px-4 py-4 active:bg-slate-50 ${
                index < profile.rows.length - 1 ? "border-b border-slate-100" : ""
              }`}
            >
              <View className="flex-row items-center gap-3">
                <View className="h-10 w-10 items-center justify-center rounded-full bg-[#E0F7F8]">
                  <MaterialCommunityIcons
                    name={rowIcon(row.icon)}
                    size={22}
                    color="#00797C"
                  />
                </View>
                <Text className="text-base font-medium text-slate-900">
                  {row.title}
                </Text>
              </View>
              <MaterialCommunityIcons name="chevron-right" size={22} color="#94a3b8" />
            </Pressable>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}
