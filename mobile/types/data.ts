export interface TabNavItem {
  id: string;
  label: string;
  icon: string;
  iconActive: string;
}

export interface FeatureItem {
  id: string;
  title: string;
  description: string;
  iconName: string;
  variant: "muted" | "primary" | "light";
  footerTrusted?: boolean;
  demoLabel?: string;
}

export interface HomeProduct {
  id: string;
  name: string;
  recommendationText: string;
  shopLabel: string;
  imageUri: string;
}

export interface ReportProduct {
  id: string;
  name: string;
  price: string;
  origin: string;
  badges: string[];
  description: string;
  imageUri: string;
  ctaLabel: string;
}

export interface BackendProduct {
  name: string;
  brand: string;
  price: number;
  country_of_origin: string;
  key_ingredients: string[];
  benefits: string;
  description: string;
}

export interface ReportTag {
  id: string;
  label: string;
  icon?: string;
}

export interface FilterChip {
  id: string;
  label: string;
}

export interface SkinData {
  app: { name: string };
  user: { displayName: string; avatarUri: string };
  navigation: { tabs: TabNavItem[] };
  analysisCapture: {
    heroTitle: string;
    heroAccent: string;
    heroSubtitle: string;
    liveStatusLabel: string;
    liveStatusDetail: string;
    conditionShort: string;
    captureTitle: string;
    captureHint: string;
    optimalLight: {
      title: string;
      description: string;
      qualityPercent: number;
    };
    lastSession: {
      title: string;
      description: string;
      hydrationDeltaPercent: number;
      sinceLabel: string;
    };
    analyzing: {
      title: string;
      description: string;
      ctaLabel: string;
    };
  };
  home: {
    badge: string;
    headline: string;
    headlineAccent: string;
    subtext: string;
    primaryCta: string;
    secondaryCta: string;
    liveAnalysis: {
      tag: string;
      title: string;
      scanningLabel: string;
      hydrationLabel: string;
      hydrationPercent: number;
      imageUri: string;
    };
    userStats: {
      trustedLabel: string;
      count: string;
      avatars: string[];
    };
    features: FeatureItem[];
    productsSection: { title: string; subtext: string };
    products: HomeProduct[];
    footerCard: { title: string; subtitle: string };
  };
  report: {
    analysis: {
      label: string;
      headline: string;
      summary: string;
      scorePercent: number;
      scoreLabel: string;
      tags: ReportTag[];
    };
    filters: {
      priceLabel: string;
      priceValue: string;
      priceOptions: string[];
      originLabel: string;
      originValue: string;
      originOptions: string[];
      veganLabel: string;
      veganOn: boolean;
      matchedCount: number;
      activeChips: FilterChip[];
    };
    products: ReportProduct[];
    medicalNote: {
      title: string;
      description: string;
      ctaLabel: string;
    };
  };
}
