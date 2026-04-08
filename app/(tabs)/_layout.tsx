import { MaterialCommunityIcons } from "@expo/vector-icons";
import { Tabs } from "expo-router";
import { View } from "react-native";
import { skinData } from "../../lib/skinData";

function TabIcon({
  name,
  color,
  focused,
}: {
  name: keyof typeof MaterialCommunityIcons.glyphMap;
  color: string;
  focused: boolean;
}) {
  return (
    <View
      className={
        focused
          ? "rounded-2xl bg-[#E0F7F8] px-3 py-1.5"
          : "px-3 py-1.5"
      }
    >
      <MaterialCommunityIcons name={name} size={22} color={color} />
    </View>
  );
}

export default function TabsLayout() {
  const [homeTab, analysisTab, profileTab] = skinData.navigation.tabs;

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: "#00797C",
        tabBarInactiveTintColor: "#94a3b8",
        tabBarStyle: {
          backgroundColor: "#ffffff",
          borderTopColor: "#e2e8f0",
          height: 72,
          paddingBottom: 10,
          paddingTop: 6,
        },
        tabBarLabelStyle: { fontSize: 12, fontWeight: "600" },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: homeTab.label,
          tabBarIcon: ({ color, focused }) => (
            <TabIcon
              name={(focused ? homeTab.iconActive : homeTab.icon) as keyof typeof MaterialCommunityIcons.glyphMap}
              color={color}
              focused={focused}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="analysis"
        options={{
          title: analysisTab.label,
          tabBarIcon: ({ color, focused }) => (
            <TabIcon
              name={
                (focused
                  ? analysisTab.iconActive
                  : analysisTab.icon) as keyof typeof MaterialCommunityIcons.glyphMap
              }
              color={color}
              focused={focused}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: profileTab.label,
          tabBarIcon: ({ color, focused }) => (
            <TabIcon
              name={
                (focused
                  ? profileTab.iconActive
                  : profileTab.icon) as keyof typeof MaterialCommunityIcons.glyphMap
              }
              color={color}
              focused={focused}
            />
          ),
        }}
      />
    </Tabs>
  );
}
