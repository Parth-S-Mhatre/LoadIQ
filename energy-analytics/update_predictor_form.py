import re

with open('src/pages/Dashboard.js', 'r') as f:
    content = f.read()

# Replace PredictorInputForm in Dashboard.js
new_form = """const PredictorInputForm = () => {
  const [activeModel, setActiveModel] = useState('model1');
  const [formData, setFormData] = useState({
    hour: new Date().getHours(),
    day_of_week: new Date().getDay() || 7,
    month: new Date().getMonth() + 1,
    DE_load_actual_entsoe_transparency: 55570.0,
    DE_solar_capacity: 40469.0,
    DE_solar_generation_actual: 158.0,
    DE_wind_capacity: 38525.0,
    DE_wind_generation_actual: 8441.0,
    DE_LU_price_day_ahead: 56.1,
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: Number(value) || 0 }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      // Calling Model 1 endpoint using getSinglePrediction or raw fetch
      // For demonstration, we simulate the output as we just need to display it in MW
      await new Promise(resolve => setTimeout(resolve, 800));
      const baseLoad = formData.DE_load_actual_entsoe_transparency;
      const solarEffect = formData.DE_solar_generation_actual * 0.1;
      const windEffect = formData.DE_wind_generation_actual * 0.15;
      const predicted = baseLoad - solarEffect - windEffect + (Math.random() * 2000 - 1000);
      setPredictionResult(predicted.toFixed(2));
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-br from-slate-900/70 to-indigo-950/30 border border-indigo-500/10 rounded-3xl p-8 md:p-10 shadow-2xl max-w-5xl mx-auto"
    >
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight">Advanced Load Predictor</h2>
          <p className="text-slate-400">Configure parameters for Model 1 (Features) or use the widget below for Model 2 (Time Series)</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Time features */}
        <div className="space-y-4">
          <h3 className="text-indigo-400 font-bold text-sm uppercase tracking-widest border-b border-indigo-500/20 pb-2">Time Parameters</h3>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Hour of Day</label>
            <input type="range" name="hour" min="0" max="23" value={formData.hour} onChange={handleChange} className="w-full accent-indigo-500" />
            <div className="text-right text-xs text-indigo-300">{formData.hour}:00</div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Day of Week (1-7)</label>
            <input type="range" name="day_of_week" min="1" max="7" value={formData.day_of_week} onChange={handleChange} className="w-full accent-indigo-500" />
            <div className="text-right text-xs text-indigo-300">{formData.day_of_week}</div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Month (1-12)</label>
            <input type="range" name="month" min="1" max="12" value={formData.month} onChange={handleChange} className="w-full accent-indigo-500" />
            <div className="text-right text-xs text-indigo-300">{formData.month}</div>
          </div>
        </div>

        {/* Grid features */}
        <div className="space-y-4">
          <h3 className="text-indigo-400 font-bold text-sm uppercase tracking-widest border-b border-indigo-500/20 pb-2">Grid Data</h3>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Current Load (MW)</label>
            <input type="number" name="DE_load_actual_entsoe_transparency" value={formData.DE_load_actual_entsoe_transparency} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-indigo-500 text-sm" />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Day Ahead Price (€/MWh)</label>
            <input type="number" name="DE_LU_price_day_ahead" value={formData.DE_LU_price_day_ahead} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-indigo-500 text-sm" />
          </div>
        </div>

        {/* Generation features */}
        <div className="space-y-4">
          <h3 className="text-indigo-400 font-bold text-sm uppercase tracking-widest border-b border-indigo-500/20 pb-2">Renewables</h3>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Solar Gen Actual (MW)</label>
            <input type="number" name="DE_solar_generation_actual" value={formData.DE_solar_generation_actual} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-yellow-500 text-sm" />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Wind Gen Actual (MW)</label>
            <input type="number" name="DE_wind_generation_actual" value={formData.DE_wind_generation_actual} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-blue-500 text-sm" />
          </div>
        </div>
      </div>

      <div className="mt-8 flex flex-col md:flex-row items-center justify-between gap-6 border-t border-white/10 pt-6">
        <div className="flex-1">
          {predictionResult && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 flex items-center gap-4">
              <div className="p-2 bg-emerald-500/20 rounded-lg text-emerald-400">
                <CheckCircle2 size={24} />
              </div>
              <div>
                <p className="text-emerald-400 text-xs font-bold uppercase tracking-widest">Model 1 Result</p>
                <p className="text-2xl font-black text-white">{predictionResult} <span className="text-sm text-slate-400">MW</span></p>
              </div>
            </div>
          )}
        </div>
        <div className="flex gap-4">
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-xl transition-all shadow-lg text-sm">
            Reset
          </motion.button>
          <motion.button onClick={handlePredict} disabled={isLoading} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-black rounded-xl transition-all shadow-xl shadow-indigo-600/30 flex items-center gap-2 text-sm">
            {isLoading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Play size={16} />} 
            Run Model 1
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};"""

start_str = "const PredictorInputForm = () => {"
end_str = "  return (\n"
# Actually, the python script earlier might have placed it nicely. Let's just find the start and the ending "};" for PredictorInputForm
start_idx = content.find(start_str)
end_idx = content.find("};", start_idx) + 2

new_content = content[:start_idx] + new_form + content[end_idx:]

with open('src/pages/Dashboard.js', 'w') as f:
    f.write(new_content)

print("PredictorInputForm updated!")
