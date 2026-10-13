import io
# io: file ke bytes ko "file jaisa object" banane ke liye, taaki pandas use padh sake.
import os
# os: Secrets / environment variables padhne ke liye (API key yahin se aati hai).

import matplotlib
# matplotlib: charts banane wali library. Pehle iska backend set karna hai.

matplotlib.use("Agg")
# "Agg" = bina screen ke chart banana. Server par screen nahi hoti, isliye ye zaruri hai.
import matplotlib.pyplot as plt
# plt: charts (figure, axes) banane ke asli functions.
import numpy as np
# numpy: numbers aur arrays ke saath kaam ke liye.
import pandas as pd
# pandas: data table (DataFrame) padhne aur analyse karne ke liye.
import requests
# requests: AI API ko internet par call karne ke liye.
import seaborn as sns
# seaborn: matplotlib ke upar sundar charts (histogram, heatmap) banane ke liye.
import streamlit as st
# streamlit: poora website/UI banata hai (upload box, tabs, buttons).

# ---------------- settings (Secrets / environment variables) ----------------
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")   # optional: enables AI insights
# Secrets se API key lo. Nahi mili to khali string "", aur AI wala tab band rahega.
AI_MODEL = os.environ.get("AI_MODEL", "claude-haiku-5-5")
# AI ka model ka naam. Secrets mein na ho to default model use hoga.
MAX_MB = 20            # max upload size
# Upload ki maximum size (MB). Bahut badi file server ko dheela ya band kar sakti hai.
MAX_AI_CALLS = 5       # AI insight requests per browser session
# Ek session mein AI ko sirf 5 baar bulane do, taaki kharcha control mein rahe.


# ---------------- core functions ----------------
@st.cache_data(show_spinner=False)
# Cache: same file dobara aaye to dobara padhna nahi padta. Isse app tez chalta hai.
def parse_file(data: bytes, name: str, sheet=None):
    # File ko bytes, naam aur (Excel ke liye) sheet ke saath padhne wala function.
    if name.lower().endswith(".csv"):
        # Naam .csv par khatam hota hai to ye CSV file hai.
        try:
            # Pehle normal (utf-8) encoding try karo.
            return pd.read_csv(io.BytesIO(data))
            # Bytes ko file banakar pandas se CSV padho aur table wapas do.
        except UnicodeDecodeError:
            # Agar encoding galat nikli (jaise purani Excel-CSV files mein hota hai)...
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
            # ...to latin-1 encoding se dobara padho, taaki file fail na ho.
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0)
    # CSV nahi hai to Excel hai. Chuni hui sheet padho, na chuni ho to pehli sheet (0).


def clean(df):
    # Missing (khali) values bharne wala function.
    """Fill missing values: median for numbers, mode for everything else."""
    # Docstring: numbers mein median aur baaki mein sabse common value bharte hain.
    d = df.copy()
    # Copy banao, taaki asli (raw) data na badle. Missing values ginti ke liye asli chahiye.
    for c in d.columns:
        # Har column ke liye ek-ek karke dekho.
        if pd.api.types.is_numeric_dtype(d[c]):
            # Agar column numbers wala hai...
            if d[c].notna().any():
                # ...aur usme kam se kam ek asli value hai (poora khali nahi)...
                d[c] = d[c].fillna(d[c].median())
                # ...to khali jagah median se bharo. Median outliers se kam bigadta hai.
        else:
            # Column text/category wala hai.
            m = d[c].mode()
            # Sabse zyada baar aane wali value nikalo.
            if len(m):
                # Agar koi mode mila (column poora khali nahi)...
                d[c] = d[c].fillna(m[0])
                # ...to khali jagah us common value se bharo.
    return d
    # Saaf kiya hua table wapas do.


def top_correlations(num, limit=5, threshold=0.5):
    # Numeric columns ke beech sabse strong jodiyan dhundhne wala function.
    num = num.iloc[:, :30]
    # Sirf pehle 30 numeric columns lo. Bahut zyada columns hone par app dheela na ho.
    if num.shape[1] < 2:
        # Jodi banane ke liye kam se kam 2 numeric columns chahiye.
        return []
        # Kam hain to khali list wapas do.
    corr = num.corr()
    # Har column ka dusre column se correlation (-1 se +1) ka table banao.
    cols = list(corr.columns)
    # Column naamon ki list.
    pairs = []
    # Yahan strong jodiyan jama hongi.
    for i in range(len(cols)):
        # Pehla column chuno.
        for j in range(i + 1, len(cols)):
            # Dusra column uske aage wala chuno, taaki har jodi sirf ek baar aaye.
            v = corr.iloc[i, j]
            # In dono columns ka correlation value.
            if pd.notna(v) and abs(v) >= threshold:
                # Value valid ho aur kam se kam threshold jitni strong ho (+ ya -) to hi lo.
                pairs.append((abs(v), cols[i], cols[j], float(v)))
                # (taakat, column1, column2, asli value) jodi list mein daalo.
    pairs.sort(reverse=True)
    # Sabse strong jodiyan sabse upar aayein.
    return [(a, b, v) for _, a, b, v in pairs[:limit]]
    # Sirf top 'limit' jodiyan (column1, column2, value) ke roop mein wapas do.


def rule_insights(raw, df):
    # Data se seedha English insights likhne wala function (AI ke bina).
    """Plain-English insights computed directly from the data (no AI needed)."""
    # Docstring: ye insights data se calculate hote hain, kisi AI ki zarurat nahi.
    out = []
    # Insights ke sentences yahan jama honge.
    rows, cols = raw.shape
    # Asli table ki rows aur columns ki ginti.
    num = df.select_dtypes(include="number")
    # Sirf numeric columns alag nikalo.
    cat = df.select_dtypes(exclude="number")
    # Baaki (text/category) columns alag nikalo.
    out.append(
        # Pehla insight: dataset ka size batao.
        f"The dataset has {rows:,} rows and {cols} columns "
        # {rows:,} number mein comma lagata hai (jaise 1,000).
        f"({num.shape[1]} numeric, {cat.shape[1]} non-numeric)."
        # Kitne numeric aur kitne non-numeric columns hain.
    )

    dup = int(raw.duplicated().sum())
    # Kitni rows bilkul same (duplicate) hain, uski ginti.
    if dup:
        # Agar kam se kam ek duplicate hai...
        out.append(f"{dup:,} duplicate rows were found ({dup / rows:.1%}); consider removing them.")
        # ...to batao kitne hain, kitna percent hain, aur hatane ki salaah do.

    miss = raw.isnull().mean().sort_values(ascending=False)
    # Har column mein kitna fraction khali hai, aur sabse zyada khali wala sabse upar.
    miss = miss[miss > 0]
    # Sirf wo columns rakho jinme kuch to khali hai.
    if miss.empty:
        # Agar koi column khali nahi hai...
        out.append("There are no missing values.")
        # ...to ye batao.
    else:
        # Warna missing values ki detail likho.
        worst = ", ".join(f"{k} ({v:.0%})" for k, v in miss.head(3).items())
        # Sabse zyada khali wale 3 columns ka "naam (percent)" text banao.
        out.append(f"{len(miss)} columns have missing values; the most affected are: {worst}.")
        # Kitne columns mein khali values hain aur sabse bura haal kiska hai, ye likho.
        heavy = miss[miss > 0.5]
        # Wo columns jinme aadhe se zyada data khali hai.
        if len(heavy):
            # Agar aise columns hain...
            out.append(
                # ...to chetavani likho.
                "Columns with more than 50% missing data ("
                + ", ".join(map(str, heavy.index))
                # Un columns ke naam comma se jodke dikhao.
                + ") are of limited use unless they can be filled reliably."
                # Batao ki ye kaam ke nahi jab tak bharose se bhare na jaayein.
            )

    const = [c for c in raw.columns if raw[c].nunique(dropna=True) <= 1]
    # Wo columns jinme sirf ek value hai ya kuch hai hi nahi. Asli data par check karte hain, bhare hue par nahi.
    if const:
        # Agar aise columns mile...
        out.append("Columns with a single value or no data at all (no information): " + ", ".join(map(str, const)) + ".")
        # ...to unke naam batao, kyunki unse koi jaankari nahi milti.

    # skew and outliers
    skewed, outl = [], []
    # Do khali lists: ek skewed columns ke liye, ek outliers wale columns ke liye.
    for c in num.columns[:30]:
        # Pehle 30 numeric columns ke liye ek-ek karke dekho.
        s = num[c]
        # Is column ki values.
        if s.nunique() <= 2:
            # Agar sirf 1-2 alag values hain (jaise 0/1 wala column)...
            continue
            # ...to skew/outlier ka matlab nahi banta, agle column par jao.
        sk = s.skew()
        # Skewness: values ek taraf kitni jhuki hain (0 ke paas = balanced).
        if pd.notna(sk) and abs(sk) > 1:
            # Agar skew valid hai aur 1 se zyada (ya -1 se kam) hai, yaani kaafi jhuka hua...
            skewed.append((abs(sk), c, sk))
            # ...to (taakat, naam, asli skew) list mein daalo.
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        # Pehla (25%) aur teesra (75%) quartile nikalo.
        iqr = q3 - q1
        # IQR = beech ke 50% data ka phailav.
        if iqr > 0:
            # Agar phailav zero nahi hai (warna outlier nikalna bekaar hai)...
            share = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
            # Wo values jo normal range se bahut bahar hain (1.5*IQR rule), unka fraction.
            if share > 0.05:
                # Agar 5% se zyada values outlier hain...
                outl.append((share, c))
                # ...to (fraction, naam) list mein daalo.
    skewed.sort(reverse=True)
    # Sabse zyada skewed sabse upar.
    outl.sort(reverse=True)
    # Sabse zyada outliers wale sabse upar.
    if skewed:
        # Agar skewed columns mile...
        txt = ", ".join(f"{c} ({'right' if sk > 0 else 'left'}-skewed)" for _, c, sk in skewed[:3])
        # Top 3 ka text: skew positive ho to right-skewed, negative ho to left-skewed.
        out.append(f"Strongly skewed distributions: {txt}.")
        # Ye insight jodo.
    if outl:
        # Agar outlier wale columns mile...
        txt = ", ".join(f"{c} ({share:.0%})" for share, c in outl[:3])
        # Top 3 ka "naam (percent)" text banao.
        out.append(f"Columns with many outliers (more than 5% of values): {txt}.")
        # Ye insight jodo.

    for a, b, v in top_correlations(num, limit=3):
        # Top 3 strong correlation jodiyan lo.
        kind = "positively" if v > 0 else "negatively"
        # Value positive ho to saath badhte hain, negative ho to ulta.
        strength = "strongly" if abs(v) >= 0.7 else "moderately"
        # 0.7 ya usse upar ho to "strongly", warna "moderately".
        out.append(f"{a} and {b} are {strength} {kind} correlated (r = {v:.2f}).")
        # English sentence banao, jaise "Fare and Pclass are moderately negatively correlated".

    # categorical
    for c in cat.columns[:20]:
        # Pehle 20 text columns ke liye ek-ek karke dekho.
        n = df[c].nunique()
        # Is column mein kitni alag-alag values hain.
        if n > 0.5 * rows and rows > 20:
            # Agar aadhe se zyada rows mein alag value hai (jaise Name ya ID)...
            out.append(f"'{c}' has {n:,} unique values, so it looks like an identifier rather than a category.")
            # ...to ye category nahi, pehchaan wala column lagta hai, ye batao.
        elif 1 < n <= 30:
            # Warna agar thodi si (2 se 30) values hain, to ye sach mein category hai.
            vc = df[c].value_counts(normalize=True)
            # Har value ka fraction, sabse zyada wala sabse upar.
            if vc.iloc[0] >= 0.6:
                # Agar sabse common value 60% ya usse zyada rows mein hai...
                out.append(f"In '{c}', the value '{vc.index[0]}' dominates ({vc.iloc[0]:.0%} of rows).")
                # ...to batao ki ek hi value haavi hai.
    return out
    # Saare insights ki list wapas do.


def build_ai_prompt(raw, df):
    # AI ko bhejne ke liye sawal (prompt) banane wala function.
    num = df.select_dtypes(include="number")
    # Numeric columns alag nikalo.
    parts = [
        # Prompt ke tukde ek list mein jama karte hain.
        f"Shape: {raw.shape[0]} rows, {raw.shape[1]} columns",
        # Table ka size.
        "Column types:\n" + raw.dtypes.astype(str).to_string(),
        # Har column ka type (number, text, ...).
        "Missing values per column:\n" + raw.isnull().sum().to_string(),
        # Har column mein kitni values khali hain.
        "Summary statistics:\n" + df.describe(include="all").to_string(),
        # Poora statistical summary (mean, min, max, count wagairah).
    ]
    corr = top_correlations(num, limit=8, threshold=0.3)
    # AI ke liye thodi halki threshold (0.3) par top 8 correlations nikalo.
    if corr:
        # Agar kuch mile...
        parts.append("Top correlations:\n" + "\n".join(f"{a} vs {b}: {v:.2f}" for a, b, v in corr))
        # ...to unhe bhi tukdon mein jodo.
    summary = "\n\n".join(parts)[:12000]
    # Sab tukde jodke 12000 akshar par kaat do, taaki prompt bahut lamba aur mehnga na ho.
    return (
        # Final prompt wapas do.
        "You are a data analyst. Below is a statistical summary of a dataset. "
        # AI ko batao ki wo data analyst hai.
        "Write clear insights in English using short bullet points under these headings: "
        # English mein, chhote bullet points mein likhne ko kaho.
        "Overview, Data quality, Key patterns, Relationships, Recommendations. "
        # Kin headings ke neeche likhna hai.
        "Only use numbers that appear in the summary. Do not invent facts.\n\n" + summary
        # Kewal summary ke numbers use karne ko kaho (jhoothe numbers se bachne ke liye), phir summary jodo.
    )


def ai_insights(prompt):
    # AI model ko prompt bhejkar jawab laane wala function.
    r = requests.post(
        # Internet par POST request bhejo.
        "https://api.anthropic.com/v1/messages",
        # AI service ka address.
        headers={
            # Request ke saath bheji jaane wali pehchaan.
            "x-api-key": API_KEY,
            # Aapki secret key, jisse service aapko pehchanti hai. Ye code mein nahi, Secrets mein rehti hai.
            "anthropic-version": "2023-06-01",
            # API ka version, jo service ko batana zaruri hai.
            "content-type": "application/json",
            # Hum JSON format mein data bhej rahe hain.
        },
        json={
            # Request ka asli data.
            "model": AI_MODEL,
            # Kaunsa model use karna hai.
            "max_tokens": 1200,
            # Jawab ki maximum lambai. Isse kharcha bhi seema mein rehta hai.
            "messages": [{"role": "user", "content": prompt}],
            # User ki taraf se hamara prompt.
        },
        timeout=120,
        # 120 second tak jawab ka intezaar karo, phir haar maan lo.
    )
    if r.status_code != 200:
        # 200 ka matlab sab theek. Isse alag kuch aaye to error hai.
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        # Error ko code aur chhote message ke saath upar bhejo, taaki UI dikha sake.
    return "".join(b.get("text", "") for b in r.json().get("content", []))
    # Jawab ke saare text hisse jodke ek string banao aur wapas do.


# ---------------- charts ----------------
def hist_grid(num):
    # Numeric columns ke histograms ek hi chart mein grid ki tarah banane wala function.
    cols = list(num.columns[:12])
    # Sirf pehle 12 columns, taaki page bahut lamba na ho.
    n = len(cols)
    # Kitne charts banane hain.
    ncols = 3
    # Ek row mein 3 charts.
    nrows = (n + ncols - 1) // ncols
    # Kitni rows chahiye (upar ki taraf round karke).
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    # Poora figure aur uske andar charts ki grid banao. Height rows ke hisaab se badhti hai.
    axes = np.array(axes).reshape(-1)
    # Grid ko ek seedhi list mein badlo, taaki ek-ek karke chala sakein.
    for ax, c in zip(axes, cols):
        # Har chart box ko ek column ke saath jodo.
        sns.histplot(num[c], bins=30, kde=bool(num[c].nunique() > 1), ax=ax, color="steelblue")
        # Histogram banao (30 dibbe). KDE curve tabhi jab column mein ek se zyada alag value ho, warna error aata hai.
        ax.set_title(str(c))
        # Chart ke upar column ka naam.
        ax.set_xlabel("")
        # Neeche ka label hata do, kyunki title mein naam pehle se hai.
    for ax in axes[n:]:
        # Jo extra box bach gaye (khali)...
        ax.axis("off")
        # ...unhe chhupa do.
    fig.tight_layout()
    # Charts ko is tarah set karo ki ek dusre par na chadhein.
    return fig
    # Tayyar figure wapas do.


def heatmap(num):
    # Correlation heatmap banane wala function.
    corr = num.iloc[:, :30].corr()
    # Pehle 30 numeric columns ka correlation table.
    size = max(6, min(14, 0.6 * len(corr)))
    # Chart ka size column ginti ke hisaab se, par 6 se 14 ke beech.
    fig, ax = plt.subplots(figsize=(size, size * 0.7))
    # Figure banao.
    sns.heatmap(corr, annot=len(corr) <= 12, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    # Heatmap banao. Numbers sirf tab likhe jab 12 ya kam columns hon, warna bheed ho jaati hai.
    ax.set_title("Correlation heatmap")
    # Chart ka title.
    fig.tight_layout()
    # Layout theek karo.
    return fig
    # Figure wapas do.


def missing_chart(raw):
    # Missing values ka bar chart banane wala function.
    miss = (raw.isnull().mean() * 100).sort_values(ascending=False)
    # Har column ka khali percent, sabse zyada upar.
    miss = miss[miss > 0].head(20)
    # Sirf jinme khali hai, aur top 20 tak.
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(miss))))
    # Figure banao. Height bars ki ginti ke hisaab se badhti hai.
    sns.barplot(x=miss.values, y=miss.index.astype(str), ax=ax, color="tomato")
    # Horizontal bar chart. Laal rang kyunki ye problem dikhata hai.
    ax.set_xlabel("% missing")
    # X axis ka label.
    ax.set_title("Missing values by column")
    # Chart ka title.
    fig.tight_layout()
    # Layout theek karo.
    return fig
    # Figure wapas do.


def category_chart(df, col):
    # Ek category column ke values ginne wala bar chart banane wala function.
    vc = df[col].value_counts().head(15)
    # Har value kitni baar aayi, top 15.
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(vc))))
    # Figure banao. Height bars ke hisaab se.
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="seagreen")
    # Horizontal bar chart, hare rang mein.
    ax.set_title(f"{col}")
    # Chart ke upar column ka naam.
    ax.set_xlabel("count")
    # X axis ka label.
    fig.tight_layout()
    # Layout theek karo.
    return fig
    # Figure wapas do.


# ---- UI ----
st.set_page_config(page_title="Auto EDA", layout="wide")
# Browser tab ka naam aur page ko poori chaudai (wide) mein dikhao. Ye sabse pehla Streamlit command hona chahiye.
st.title("Automated Exploratory Data Analysis")
# Page ka bada heading.
st.caption("Upload a CSV or Excel file to get a summary, insights in English, and charts automatically.")
# Heading ke neeche chhota text, jo batata hai ki app kya karta hai.

up = st.file_uploader("Upload a CSV or Excel (.xlsx) file", type=["csv", "xlsx"])
# Upload box. Sirf .csv aur .xlsx files chunne denge, baaki type reject ho jayenge.
if up is None:
    # Agar abhi tak koi file upload nahi hui...
    st.info("Waiting for a file...")
    # ...to ye message dikhao.
    st.stop()
    # Aage ka code mat chalao, jab tak file na aaye.

data = up.getvalue()
# Upload hui file ke saare bytes lo.
if len(data) > MAX_MB * 1024 * 1024:
    # Agar file ki size 20 MB se zyada hai...
    st.error(f"File is larger than {MAX_MB} MB. Please upload a smaller file.")
    # ...to error dikhao.
    st.stop()
    # Aur ruk jao, taaki server par bojh na pade.

sheet = None
# Shuru mein koi sheet nahi chuni.
if up.name.lower().endswith(".xlsx"):
    # Agar file Excel hai...
    sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
    # ...to uski saari sheets ke naam nikalo.
    if len(sheets) > 1:
        # Agar ek se zyada sheets hain...
        sheet = st.selectbox("Sheet", sheets)
        # ...to user se poocho ki kaunsi sheet analyse karni hai.

try:
    # File padhne ki koshish karo.
    raw = parse_file(data, up.name, sheet)
    # Asli (raw) table padho. Isme khali values waisi hi rehti hain.
except Exception as e:
    # Agar file kharab ya galat format ki ho...
    st.error(f"Could not read this file: {e}")
    # ...to saaf error dikhao, app crash na ho.
    st.stop()
    # Aur ruk jao.

if raw.empty:
    # Agar table mein ek bhi row nahi...
    st.warning("The file has no rows.")
    # ...to chetavani do.
    st.stop()
    # Aur ruk jao.

df = clean(raw)
# Khali values bharke saaf table banao (charts aur statistics ke liye).
num = df.select_dtypes(include="number")
# Sirf numeric columns alag nikalo.
cat_cols = [
    # Wo text columns jinke charts banane layak hain.
    c for c in df.columns
    # Har column ko dekho...
    if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 30
    # ...jo numeric nahi hai aur jisme 2 se 30 alag values hain.
]
insights = rule_insights(raw, df)
# Automatic English insights nikalo.

st.success(f"Data loaded: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
# Hari patti mein dikhao ki file load ho gayi aur kitni rows/columns hain.
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Insights", "Charts", "AI Insights"])
# Chaar tabs banao. Isse page lamba nahi hota aur sab kuch saaf dikhta hai.

with tab1:
    # Pehle tab (Summary) ke andar ka content.
    st.subheader("Preview")
    # Section ka heading.
    st.dataframe(raw.head(20), use_container_width=True)
    # Asli table ki pehli 20 rows dikhao, poori chaudai mein.
    st.subheader("Summary statistics")
    # Section ka heading.
    st.dataframe(df.describe(include="all").T, use_container_width=True)
    # Har column ka statistical summary. .T se table palat jaati hai, jisse padhna aasan hota hai.
    st.subheader("Missing values")
    # Section ka heading.
    mv = pd.DataFrame({"missing": raw.isnull().sum(), "percent": (raw.isnull().mean() * 100).round(1)})
    # Har column ke khali values ki ginti aur percent ka chhota table.
    st.dataframe(mv, use_container_width=True)
    # Wo table dikhao.
    st.caption("Missing values are filled (median for numbers, most common value for text) for the charts and statistics.")
    # User ko batao ki charts aur statistics ke liye khali values bhari gayi hain, taaki wo confuse na ho.

with tab2:
    # Doosre tab (Insights) ke andar ka content.
    st.subheader("Key insights")
    # Section ka heading.
    for line in insights:
        # Har insight ko ek-ek karke lo.
        st.markdown(f"- {line}")
        # Bullet point ke roop mein dikhao.
    st.download_button(
        # Insights ko file mein download karne ka button.
        "Download insights (.txt)",
        # Button par likha text.
        "\n".join(f"- {x}" for x in insights),
        # Download hone wali file ka content: har insight naye line par.
        file_name="eda_insights.txt",
        # Download hone wali file ka naam.
    )

with tab3:
    # Teesre tab (Charts) ke andar ka content.
    if raw.isnull().any().any():
        # Agar poore table mein kahin bhi khali value hai...
        st.pyplot(missing_chart(raw))
        # ...to missing values ka chart dikhao.
    if num.shape[1] > 0:
        # Agar kam se kam ek numeric column hai...
        st.subheader("Distributions of numeric columns")
        # Section ka heading.
        st.pyplot(hist_grid(num))
        # Histograms ki grid dikhao.
    if num.shape[1] > 1:
        # Heatmap ke liye kam se kam 2 numeric columns chahiye.
        st.subheader("Correlations")
        # Section ka heading.
        st.pyplot(heatmap(num))
        # Correlation heatmap dikhao.
    if cat_cols:
        # Agar chart layak text columns hain...
        st.subheader("Categorical columns")
        # Section ka heading.
        cols = st.columns(2)
        # Page ko 2 column mein baanto, taaki charts side-by-side aayein.
        for i, c in enumerate(cat_cols[:8]):
            # Pehle 8 category columns ke liye (i = number, c = naam).
            with cols[i % 2]:
                # Ek chart baayein, agla daayein (i % 2 ek-ek karke badalta hai).
                st.pyplot(category_chart(df, c))
                # Is category ka bar chart dikhao.
    plt.close("all")
    # Saare banaye hue charts memory se hata do, warna server ki memory bharti jaati hai.

with tab4:
    # Chauthe tab (AI Insights) ke andar ka content.
    if not API_KEY:
        # Agar API key set nahi hai...
        st.info("AI insights are not enabled on this app. The Insights tab already shows automatic insights.")
        # ...to batao ki AI band hai, par Insights tab mein automatic insights pehle se hain.
    else:
        # Key set hai, to AI ka button dikhao.
        st.caption("Only summary statistics (not your full file) are sent to the AI model.")
        # Privacy ke liye batao ki poori file nahi, sirf summary AI ko jaati hai.
        if "ai_calls" not in st.session_state:
            # Agar is session mein ginti abhi shuru nahi hui...
            st.session_state.ai_calls = 0
            # ...to zero se shuru karo.
        if st.button("Generate AI insights"):
            # Agar user ne button dabaya...
            if st.session_state.ai_calls >= MAX_AI_CALLS:
                # ...aur limit poori ho chuki hai...
                st.warning("AI insight limit reached for this session. Refresh the page to continue.")
                # ...to chetavani do.
            else:
                # Limit baaki hai, to AI ko bulao.
                st.session_state.ai_calls += 1
                # Ginti ek badhao.
                with st.spinner("Analysing..."):
                    # "Analysing..." ghoomta hua dikhao jab tak jawab aaye.
                    try:
                        # AI call karne ki koshish karo.
                        st.markdown(ai_insights(build_ai_prompt(raw, df)))
                        # Prompt banao, AI ko bhejo, aur jawab dikhao.
                    except Exception as e:
                        # Agar AI call fail ho (key galat, net dikkat, ...)...
                        st.error(f"Could not get AI insights: {e}")
                        # ...to error dikhao, app crash na ho.
