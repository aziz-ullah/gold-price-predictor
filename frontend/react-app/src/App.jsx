import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
    ResponsiveContainer,
    LineChart,
    Line,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
} from "recharts";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:8000";
const TOLA_CONVERSION = 0.375;

const round2 = (value) =>
    typeof value === "number" && Number.isFinite(value)
        ? value.toFixed(2)
        : "-";

const formatPercent = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "-";
    }
    const val = Number(value).toFixed(2);
    return Number(value) > 0 ? `+${val}%` : `${val}%`;
};

function App() {
    const [latest, setLatest] = useState(null);
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError("");
        try {
            const [latestResponse, historyResponse] = await Promise.all([
                axios.get(`${API_BASE_URL}/predict/latest`),
                axios.get(`${API_BASE_URL}/history`, { params: { limit: 30 } }),
            ]);
            setLatest(latestResponse.data);
            setHistory(historyResponse.data?.data ?? []);
        } catch (err) {
            const message = err.response?.data?.detail || err.message || "Unknown error";
            setError(message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
    }, [fetchData]);

    const viewModel = useMemo(() => {
        if (!latest) {
            return null;
        }

        const todayOz = Number(latest.today_price ?? NaN);
        const tomorrowOz = Number(latest.predicted_tomorrow_price ?? NaN);
        const change = Number(latest.predicted_change_percent ?? NaN);
        const usdToPkr = Number(latest.usd_to_pkr_rate ?? NaN);
        const successProbability = Number(latest.success_probability ?? NaN);

        const todayDate = latest.today_date
            ? new Date(latest.today_date).toLocaleDateString()
            : latest.date
                ? new Date(latest.date).toLocaleDateString()
                : "-";
        const tomorrowDate = latest.tomorrow_date
            ? new Date(latest.tomorrow_date).toLocaleDateString()
            : latest.date
                ? new Date(latest.date).toLocaleDateString()
                : "-";

        const tola = (value) => value * TOLA_CONVERSION;
        const toPkr = (value) => value * usdToPkr;

        const todayRow = {
            label: "Today",
            date: todayDate,
            usdTola: round2(tola(todayOz)),
            pkrTola: round2(toPkr(tola(todayOz))),
            changeText: "-",
            changeClass: "",
            probability: null,
        };

        const tomorrowRow = {
            label: "Tomorrow (Predicted)",
            date: tomorrowDate,
            usdTola: round2(tola(tomorrowOz)),
            pkrTola: round2(toPkr(tola(tomorrowOz))),
            changeText: formatPercent(change),
            changeClass:
                Number.isFinite(change) && change > 0
                    ? "positive"
                    : Number.isFinite(change) && change < 0
                        ? "negative"
                        : "",
            probability: Number.isFinite(successProbability) ? successProbability : null,
        };

        return {
            rows: [todayRow, tomorrowRow],
            usdToPkr: Number.isFinite(usdToPkr) ? round2(usdToPkr) : "-",
        };
    }, [latest]);

    const chartData = useMemo(() => {
        const base = (history ?? []).map((entry) => {
            const isoString = entry.date ?? null;
            const dateObj = isoString ? new Date(isoString) : null;
            const label = dateObj
                ? dateObj.toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                })
                : "-";
            return {
                iso: dateObj ? dateObj.getTime() : Number.NEGATIVE_INFINITY,
                date: label,
                price: Number(entry.price ?? NaN),
            };
        });

        const latestIso = latest?.today_date ?? latest?.date ?? null;
        const latestPrice = Number(latest?.today_price ?? NaN);

        if (latestIso && Number.isFinite(latestPrice)) {
            const latestDateObj = new Date(latestIso);
            const latestLabel = latestDateObj.toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
            });
            const latestTime = latestDateObj.getTime();

            const existingIndex = base.findIndex((row) => row.iso === latestTime);
            if (existingIndex >= 0) {
                base[existingIndex].price = latestPrice;
                base[existingIndex].date = latestLabel;
            } else {
                base.push({ iso: latestTime, date: latestLabel, price: latestPrice });
            }
        }

        const sorted = base.sort((a, b) => a.iso - b.iso);
        return sorted.map(({ date, price }) => ({ date, price }));
    }, [history, latest]);

    return (
        <div className="page">
            <div className="container">
                <header className="header">
                    <span className="icon" role="img" aria-label="gold">
                        💰
                    </span>
                    <h1>Gold Price Predictor</h1>
                    <p className="subhead">
                        Exchange Rate (USD → PKR): ₨{viewModel?.usdToPkr ?? "-"}
                    </p>
                </header>

                <div className="actions">
                    <button onClick={fetchData} disabled={loading}>
                        {loading ? "Refreshing..." : "Refresh"}
                    </button>
                </div>

                {error && <div className="error">{error}</div>}

                <div className="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Period</th>
                                <th>Date</th>
                                <th>USD / tola</th>
                                <th>PKR / tola</th>
                                <th>Predicted Change</th>
                                <th>Success Probability</th>
                            </tr>
                        </thead>
                        <tbody>
                            {(viewModel?.rows ?? []).map((row) => (
                                <tr key={row.label}>
                                    <td>{row.label}</td>
                                    <td>{row.date}</td>
                                    <td>${row.usdTola}</td>
                                    <td>₨{row.pkrTola}</td>
                                    <td>
                                        <span className={`change-pill ${row.changeClass}`}>{row.changeText}</span>
                                    </td>
                                    <td>
                                        {row.probability === null ? (
                                            <span className="muted">-</span>
                                        ) : (
                                            <div className="probability">
                                                <div className="probability-bar">
                                                    <div
                                                        className="probability-fill"
                                                        style={{ width: `${Math.min(100, Math.max(0, row.probability))}%` }}
                                                    />
                                                </div>
                                                <span>{round2(row.probability)}%</span>
                                            </div>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <section className="chart-section">
                    <h2>Recent Gold Price Trend (USD / oz)</h2>
                    <div className="chart-card">
                        {chartData.length === 0 ? (
                            <div className="chart-empty">No historical data available.</div>
                        ) : (
                            <ResponsiveContainer width="100%" height={320}>
                                <LineChart data={chartData} margin={{ top: 16, right: 24, left: 0, bottom: 8 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                    <XAxis dataKey="date" tick={{ fontSize: 12 }} minTickGap={16} />
                                    <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                                    <Tooltip
                                        formatter={(value) => [`$${round2(Number(value))}`, "Price"]}
                                        labelFormatter={(label) => `Date: ${label}`}
                                    />
                                    <Line type="monotone" dataKey="price" stroke="#2563eb" strokeWidth={2} dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </section>
            </div>
        </div>
    );
}

export default App;

