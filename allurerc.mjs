import { readFileSync } from "node:fs";

import { defineConfig } from "allure";

const legacyCategories = JSON.parse(
  readFileSync(new URL("./allure/categories.json", import.meta.url), "utf8"),
);

const categoryRules = legacyCategories.map((category, index) => ({
  id: `agent-team-${index + 1}`,
  name: category.name,
  matchers: {
    statuses: category.matchedStatuses,
    ...(category.messageRegex
      ? { message: new RegExp(category.messageRegex, "i") }
      : {}),
    ...(category.traceRegex
      ? { trace: new RegExp(category.traceRegex, "i") }
      : {}),
  },
}));

export default defineConfig({
  name: "Agent Team test report",
  output: "./allure-report",
  categories: {
    rules: categoryRules,
  },
  qualityGate: {
    rules: [
      {
        maxFailures: 0,
        successRate: 1,
        minTestsCount: 1,
      },
    ],
  },
  plugins: {
    awesome: {
      options: {
        reportName: "Agent Team test report",
        reportLanguage: "en",
        singleFile: false,
        open: false,
        publish: false,
      },
    },
  },
});
