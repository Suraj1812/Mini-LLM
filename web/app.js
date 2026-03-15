const statusTitle = document.querySelector("#status-title");
const statusMessage = document.querySelector("#status-message");
const statusReady = document.querySelector("#status-ready");
const statusLoss = document.querySelector("#status-loss");
const statusTrainedAt = document.querySelector("#status-trained-at");
const statusError = document.querySelector("#status-error");
const trainForm = document.querySelector("#train-form");
const generateForm = document.querySelector("#generate-form");
const trainButton = document.querySelector("#train-button");
const generateButton = document.querySelector("#generate-button");
const output = document.querySelector("#output");
const trainingText = document.querySelector("#training-text");

let pollingHandle = null;

async function fetchStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) {
    throw new Error("Unable to load backend status.");
  }

  return response.json();
}

function renderStatus(status) {
  const titleMap = {
    idle: "Waiting for training",
    running: "Training in progress",
    ready: "Model ready",
    error: "Action needed",
  };

  statusTitle.textContent = titleMap[status.status] || "Unknown state";
  statusMessage.textContent = status.message || "No backend message available.";
  statusReady.textContent = status.model_ready ? "Yes" : "No";
  statusLoss.textContent =
    typeof status.last_loss === "number" ? status.last_loss.toFixed(4) : "-";
  statusTrainedAt.textContent = status.trained_at || "-";
  statusError.hidden = !status.last_error;
  statusError.textContent = status.last_error || "";
  trainButton.disabled = status.status === "running";
  generateButton.disabled = !status.model_ready || status.status === "running";
}

async function refreshStatus() {
  try {
    renderStatus(await fetchStatus());
  } catch (error) {
    statusTitle.textContent = "Connection issue";
    statusMessage.textContent = error.message;
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }

  return data;
}

trainForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = trainingText.value.trim();
  const epochs = Number(document.querySelector("#epochs").value || 5);

  trainButton.disabled = true;
  output.textContent = "Training started. Backend status will update automatically.";

  try {
    await postJson("/api/train", { text, epochs });
    await refreshStatus();
  } catch (error) {
    output.textContent = error.message;
    trainButton.disabled = false;
  }
});

generateForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const prompt = document.querySelector("#prompt").value.trim();
  const length = Number(document.querySelector("#length").value || 50);

  generateButton.disabled = true;
  output.textContent = "Generating...";

  try {
    const result = await postJson("/api/generate", { prompt, length });
    output.textContent = result.output;
  } catch (error) {
    output.textContent = error.message;
  } finally {
    await refreshStatus();
  }
});

async function bootstrap() {
  await refreshStatus();
  pollingHandle = window.setInterval(refreshStatus, 2000);
}

bootstrap();

window.addEventListener("beforeunload", () => {
  if (pollingHandle) {
    window.clearInterval(pollingHandle);
  }
});
