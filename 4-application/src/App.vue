<template>
  <div class="page">
    <header>
      <p class="eyebrow">4 — application</p>
      <h1>House price visualizer</h1>
      <p class="lede">
        Enter house details and call the serving API to predict price.
      </p>
      <p class="api-status" :class="apiOk ? 'ok' : 'down'">
        API {{ apiOk ? "connected" : apiMessage }}
      </p>
    </header>

    <main>
      <form class="card form" @submit.prevent="predict">
        <h2>House</h2>

        <label>
          Square feet
          <input v-model.number="form.sqft" type="number" min="1" required />
        </label>
        <label>
          Bedrooms
          <input v-model.number="form.bedrooms" type="number" min="0" step="1" required />
        </label>
        <label>
          Bathrooms
          <input v-model.number="form.bathrooms" type="number" min="0" step="0.5" required />
        </label>
        <label>
          Year built
          <input v-model.number="form.year_built" type="number" min="1800" max="2026" required />
        </label>
        <label>
          Location
          <select v-model="form.location">
            <option v-for="loc in locations" :key="loc" :value="loc">{{ loc }}</option>
          </select>
        </label>
        <label>
          Condition
          <select v-model="form.condition">
            <option v-for="cond in conditions" :key="cond" :value="cond">{{ cond }}</option>
          </select>
        </label>

        <button type="submit" :disabled="loading">
          {{ loading ? "Predicting…" : "Predict price" }}
        </button>
        <p v-if="error" class="error">{{ error }}</p>
      </form>

      <section class="card result">
        <h2>Prediction</h2>
        <p v-if="pricePred === null" class="placeholder">
          Submit the form to see the predicted price.
        </p>
        <template v-else>
          <p class="price">{{ formattedPrice }}</p>
          <dl>
            <div><dt>Size</dt><dd>{{ form.sqft.toLocaleString() }} sqft</dd></div>
            <div><dt>Beds / baths</dt><dd>{{ form.bedrooms }} / {{ form.bathrooms }}</dd></div>
            <div><dt>Built</dt><dd>{{ form.year_built }}</dd></div>
            <div><dt>Location</dt><dd>{{ form.location }}</dd></div>
            <div><dt>Condition</dt><dd>{{ form.condition }}</dd></div>
          </dl>
        </template>
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";

const locations = ["Suburb", "Downtown", "Rural", "Urban", "Waterfront", "Mountain"];
const conditions = ["Poor", "Fair", "Good", "Excellent"];

const form = reactive({
  sqft: 1527,
  bedrooms: 2,
  bathrooms: 1.5,
  location: "Suburb",
  year_built: 1956,
  condition: "Good",
});

const loading = ref(false);
const error = ref("");
const pricePred = ref(null);
const apiOk = ref(false);
const apiMessage = ref("not connected");

const formattedPrice = computed(() =>
  pricePred.value === null
    ? ""
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(pricePred.value)
);

async function checkHealth() {
  try {
    const response = await fetch("/health");
    apiOk.value = response.ok;
    apiMessage.value = response.ok ? "connected" : "unavailable";
  } catch {
    apiOk.value = false;
    apiMessage.value = "unavailable — start 3-serving";
  }
}

async function predict() {
  loading.value = true;
  error.value = "";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    if (!response.ok) {
      throw new Error(`API error ${response.status}`);
    }
    const data = await response.json();
    pricePred.value = data.price_pred;
  } catch (err) {
    error.value = err.message || "Could not reach the API.";
    pricePred.value = null;
  } finally {
    loading.value = false;
  }
}

onMounted(checkHealth);
</script>

<style>
:root {
  color-scheme: light;
  --bg: #f4efe6;
  --ink: #1c1917;
  --muted: #57534e;
  --card: #fffcf7;
  --line: #e7e0d4;
  --accent: #0f766e;
  --accent-ink: #f0fdfa;
  --bad: #b91c1c;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--ink);
}

.page {
  max-width: 960px;
  margin: 0 auto;
  padding: 2.5rem 1.25rem 4rem;
}

header {
  margin-bottom: 2rem;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
  color: var(--accent);
  margin: 0 0 0.35rem;
}

h1,
h2 {
  margin: 0 0 0.5rem;
}

.lede,
.placeholder {
  color: var(--muted);
}

.api-status {
  display: inline-block;
  margin: 0.75rem 0 0;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.85rem;
  background: #fecaca;
  color: var(--bad);
}

.api-status.ok {
  background: #ccfbf1;
  color: var(--accent);
}

main {
  display: grid;
  gap: 1.25rem;
}

@media (min-width: 800px) {
  main {
    grid-template-columns: 1fr 1fr;
    align-items: start;
  }
}

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.25rem 1.35rem 1.5rem;
}

.form {
  display: grid;
  gap: 0.85rem;
}

label {
  display: grid;
  gap: 0.3rem;
  font-size: 0.9rem;
  color: var(--muted);
}

input,
select,
button {
  font: inherit;
  color: var(--ink);
}

input,
select {
  width: 100%;
  padding: 0.55rem 0.65rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}

button {
  margin-top: 0.35rem;
  border: 0;
  border-radius: 8px;
  padding: 0.7rem 1rem;
  background: var(--accent);
  color: var(--accent-ink);
  cursor: pointer;
}

button:disabled {
  opacity: 0.65;
  cursor: wait;
}

.error {
  color: var(--bad);
  margin: 0;
}

.price {
  font-size: 2.4rem;
  font-weight: 700;
  margin: 0.4rem 0 1rem;
}

dl {
  margin: 0;
  display: grid;
  gap: 0.45rem;
}

dl div {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  border-top: 1px solid var(--line);
  padding-top: 0.45rem;
}

dt {
  color: var(--muted);
}

dd {
  margin: 0;
}
</style>
