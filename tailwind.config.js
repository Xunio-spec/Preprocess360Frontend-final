// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        card: '#f9fafb', // light background color for cards
        'card-foreground': '#1f2937' // dark text color for card text
      },
    },
  },
};

// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
