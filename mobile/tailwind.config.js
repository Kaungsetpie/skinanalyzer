/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx,ts,tsx}", "./components/**/*.{js,jsx,ts,tsx}"],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        teal: {
          primary: "#00797C",
          dark: "#004D4D",
          alt: "#006D77",
        },
      },
    },
  },
  plugins: [],
};
