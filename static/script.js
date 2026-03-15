// static/script.js
const API_URL = "http://localhost:5000/api/ask";

// --- Global Charts ---
let winChart = null;
let timeChart = null;

document.addEventListener("DOMContentLoaded", () => {
    loadHistory();
    initCharts();
});

// --- History Logic ---
function loadHistory() {
    const history = JSON.parse(localStorage.getItem("gf_history") || "[]");
    const container = document.getElementById("historyList");
    container.innerHTML = "";

    if (history.length === 0) {
        container.innerHTML = '<p class="empty-msg">No questions yet.</p>';
        return;
    }

    history.reverse().forEach(q => {
        const div = document.createElement("div");
        div.className = "history-item";
        div.innerText = q;
        div.title = q; // Tooltip for long text
        div.onclick = () => {
            document.getElementById("questionInput").value = q;
            askQuestion();
        };
        container.appendChild(div);
    });
}

function saveHistory(question) {
    let history = JSON.parse(localStorage.getItem("gf_history") || "[]");
    if (history[history.length - 1] !== question) {
        history.push(question);
        localStorage.setItem("gf_history", JSON.stringify(history));
        loadHistory();
    }
}

function clearHistory() {
    if(confirm("Clear all history and stats?")) {
        localStorage.removeItem("gf_history");
        localStorage.removeItem("gf_wins");
        loadHistory();
        if (winChart) {
            winChart.data.datasets[0].data = [0, 0, 0];
            winChart.update();
        }
    }
}

// --- Chart Logic ---
function initCharts() {
    const wins = JSON.parse(localStorage.getItem("gf_wins") || '{"A":0, "B":0, "C":0}');
    
    // 1. Sidebar Win Chart (Doughnut)
    const ctxWin = document.getElementById('winChart').getContext('2d');
    winChart = new Chart(ctxWin, {
        type: 'doughnut',
        data: {
            labels: ['LLM', 'Vector', 'Hybrid'],
            datasets: [{
                data: [wins.A, wins.B, wins.C],
                backgroundColor: ['#e67e22', '#2980b9', '#8e44ad'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom', // Legend at bottom for sidebar fit
                    labels: { color: 'white', boxWidth: 10, font: {size: 10} }
                }
            }
        }
    });

    // 2. Header Time Chart (Bar)
    const ctxTime = document.getElementById('timeChart').getContext('2d');
    timeChart = new Chart(ctxTime, {
        type: 'bar',
        data: {
            labels: ['LLM', 'Vector', 'Hybrid'],
            datasets: [{
                label: 'Time (s)',
                data: [0, 0, 0],
                backgroundColor: ['#e67e22', '#2980b9', '#8e44ad']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }, // Hide legend to save space
            scales: {
                y: { beginAtZero: true, grid: { display: false } },
                x: { grid: { display: false } }
            }
        }
    });
}

function updateWinStats(winnerLetter) {
    let wins = JSON.parse(localStorage.getItem("gf_wins") || '{"A":0, "B":0, "C":0}');
    if(wins[winnerLetter] !== undefined) wins[winnerLetter]++;
    localStorage.setItem("gf_wins", JSON.stringify(wins));
    winChart.data.datasets[0].data = [wins.A, wins.B, wins.C];
    winChart.update();
}

function updateTimeStats(tLLM, tVec, tHyb) {
    timeChart.data.datasets[0].data = [tLLM, tVec, tHyb];
    timeChart.update();
}

// --- QA Logic ---
async function askQuestion() {
    const questionInput = document.getElementById("questionInput");
    const question = questionInput.value.trim();
    const btn = document.getElementById("askBtn");
    const loadingDiv = document.getElementById("loading");
    const resultsGrid = document.getElementById("resultsGrid");
    const winnerBox = document.getElementById("winnerBox");

    if (!question) { alert("Please enter a question."); return; }

    btn.disabled = true;
    loadingDiv.classList.remove("hidden");
    resultsGrid.classList.add("hidden");
    winnerBox.classList.add("hidden");

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) throw new Error("Server error");
        const data = await response.json();

        // Fill Results
        document.getElementById("res-llm").innerText = data.llm.text;
        document.getElementById("time-llm").innerText = data.llm.time + "s";

        document.getElementById("res-vec").innerText = data.vector.text;
        document.getElementById("time-vec").innerText = data.vector.time + "s";

        document.getElementById("res-hyb").innerText = data.hybrid.text;
        document.getElementById("time-hyb").innerText = data.hybrid.time + "s";

        document.getElementById("winnerText").innerText = data.winner_label;

        // Update Data
        saveHistory(question);
        updateWinStats(data.winner_letter);
        updateTimeStats(data.llm.time, data.vector.time, data.hybrid.time);

        // Show UI
        loadingDiv.classList.add("hidden");
        resultsGrid.classList.remove("hidden");
        winnerBox.classList.remove("hidden");

    } catch (error) {
        console.error("Error:", error);
        alert("Error fetching answer. Is server.py running?");
        loadingDiv.classList.add("hidden");
    } finally {
        btn.disabled = false;
    }
}

document.getElementById("questionInput").addEventListener("keypress", (e) => {
    if (e.key === "Enter") askQuestion();
});