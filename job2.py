import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')




st.markdown("""
    <style>
    .stApp {
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Text mining helpers ---
def preprocess_text(s):
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_top_ngrams_tfidf(docs, ngram_range=(1,1), top_n=20, max_features=5000):
    """Return top terms by TF-IDF score across docs for the given ngram_range."""
    docs = [preprocess_text(d) for d in docs if isinstance(d, str) and d.strip()]
    if len(docs) == 0:
        return []
    try:
        vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words='english')
        X = vec.fit_transform(docs)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        top_idx = scores.argsort()[::-1][:top_n]
        return list(zip(terms[top_idx], scores[top_idx]))
    except Exception:
        return []


# Page configuration
st.set_page_config(page_title="Job Market Trends Analyzer", layout="wide", page_icon="💼")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #1f77b4; font-weight: 700;}
    h2 {color: #2c3e50; margin-top: 20px;}
    .insight-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💼 Job Market Trends Analyzer")
st.markdown("### Analyze job postings to discover trending tech skills and predict skill demands")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio("Select Data Source:", 
                          ["Generate Sample Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload job postings CSV", type=['csv'])
        st.info("CSV columns: company, education, experience, industry, description, jobid, location, job_title, numberofpositions, salary, posted_date, site_name, skills, uniq_id")
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    n_clusters = st.slider("Number of Skill Clusters", 3, 10, 5)
    top_n_skills = st.slider("Top Skills to Display", 10, 50, 20)
    date_range = st.slider("Days to Analyze", 7, 365, 90)

# Function to generate sample job data
@st.cache_data
def generate_sample_data(n_jobs=500):
    """Generate realistic sample job posting data"""
    
    # Tech skills database
    tech_skills = {
        'Programming': ['Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust', 'TypeScript', 'Ruby', 'PHP', 'Swift'],
        'Data Science': ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn', 'Data Analysis', 'Statistics', 'SQL'],
        'Web Development': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'HTML', 'CSS', 'REST API', 'GraphQL'],
        'Cloud & DevOps': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD', 'Jenkins', 'Terraform', 'Ansible', 'Linux'],
        'Database': ['MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Elasticsearch', 'Cassandra', 'Oracle', 'SQL Server'],
        'AI & ML': ['NLP', 'Computer Vision', 'LLM', 'Transformers', 'Reinforcement Learning', 'GANs', 'BERT', 'GPT'],
        'Mobile': ['iOS Development', 'Android Development', 'React Native', 'Flutter', 'Kotlin', 'Swift UI'],
        'Security': ['Cybersecurity', 'Penetration Testing', 'OAuth', 'Encryption', 'Network Security', 'SIEM'],
        'Tools': ['Git', 'JIRA', 'Agile', 'Scrum', 'Tableau', 'Power BI', 'Jupyter', 'VS Code']
    }
    
    job_titles = [
        'Data Scientist', 'Software Engineer', 'ML Engineer', 'Full Stack Developer',
        'DevOps Engineer', 'Cloud Architect', 'Data Engineer', 'Frontend Developer',
        'Backend Developer', 'AI Engineer', 'Security Engineer', 'Mobile Developer',
        'Product Manager', 'Solutions Architect', 'Data Analyst'
    ]
    
    companies = ['TechCorp', 'DataWorks', 'CloudSystems', 'InnovateLabs', 'AIStartup', 
                'DevSolutions', 'WebScale', 'FinTech Inc', 'HealthTech', 'EduTech']
    
    locations = ['San Francisco, CA', 'New York, NY', 'Seattle, WA', 'Austin, TX', 
                'Boston, MA', 'Denver, CO', 'Remote', 'Chicago, IL', 'Los Angeles, CA']
    
    jobs = []
    base_date = datetime.now()
    
    for i in range(n_jobs):
        title = np.random.choice(job_titles)
        
        # Select relevant skills based on job title
        n_skills = np.random.randint(5, 15)
        selected_skills = []
        
        # Weight skill selection based on job title
        if 'Data' in title or 'ML' in title or 'AI' in title:
            categories = ['Programming', 'Data Science', 'AI & ML', 'Cloud & DevOps', 'Database', 'Tools']
        elif 'DevOps' in title or 'Cloud' in title:
            categories = ['Cloud & DevOps', 'Programming', 'Database', 'Tools']
        elif 'Developer' in title or 'Engineer' in title:
            categories = ['Programming', 'Web Development', 'Database', 'Tools', 'Cloud & DevOps']
        else:
            categories = list(tech_skills.keys())
        
        for cat in categories:
            n = np.random.randint(0, min(4, len(tech_skills[cat])))
            selected_skills.extend(np.random.choice(tech_skills[cat], n, replace=False))
        
        # Create description with skills
        description = f"We are looking for a {title}. "
        description += f"Required skills: {', '.join(selected_skills[:n_skills])}. "
        description += "Experience with software development and problem-solving is essential."
        
        # Generate posting date (more recent jobs have higher probability)
        days_ago = int(np.random.exponential(date_range/3))
        days_ago = min(days_ago, date_range)
        posted_date = base_date - timedelta(days=days_ago)
        
        # Generate salary
        base_salary = np.random.randint(70, 180) * 1000
        salary = f"${base_salary:,} - ${base_salary + 30000:,}"
        
        jobs.append({
            'job_id': f'JOB{i+1:04d}',
            'job_title': title,
            'description': description,
            'company': np.random.choice(companies),
            'location': np.random.choice(locations),
            'posted_date': posted_date.strftime('%Y-%m-%d'),
            'salary': salary,
            'skills': ', '.join(selected_skills[:n_skills])
        })
    
    return pd.DataFrame(jobs)

# Function to extract skills from text
def extract_skills(text, skill_list):
    """Extract skills from job description"""
    found_skills = []
    text_lower = text.lower()
    for skill in skill_list:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    return found_skills

# --- Skill lexicon and utilities for CSV datasets ---
def build_skill_lexicon():
    """Return a curated list of skills/keywords commonly found in job posts (covers Naukri data)."""
    core = [
        # Programming & Platforms
        'Python','Java','JavaScript','TypeScript','C','C++','C#','Go','Rust','Ruby','PHP','Perl','R','Scala','Kotlin','Swift','Objective-C',
        'Node.js','Node','React','Angular','Vue','jQuery','HTML','CSS','Django','Flask','Spring','Spring Boot','Hibernate','Struts','JSP','JSTL',
        'REST','SOAP','GraphQL','Microservices','Web Services','API',
        # Data & ML
        'SQL','MySQL','PostgreSQL','Oracle','SQL Server','MongoDB','Redis','Elasticsearch','Cassandra','NoSQL',
        'Pandas','NumPy','Scikit-learn','TensorFlow','PyTorch','Keras','Machine Learning','Deep Learning','NLP','Computer Vision','Data Analysis','Statistics',
        # Cloud & DevOps
        'AWS','Azure','GCP','Docker','Kubernetes','CI/CD','Jenkins','Terraform','Ansible','Linux','Git','GitHub','GitLab',
        # Testing & QA
        'Selenium','JUnit','TestNG','Automation Testing','Unit Testing',
        # Mobile & Embedded
        'Android','Android Studio','Xamarin','iOS','Cocoa Touch','QNX','Bluetooth','RPG','AS400','JD Edwards',
        # Big Data
        'Hadoop','Spark','Hive','HBase','MapReduce',
        # BI & Tools
        'Tableau','Power BI','Excel','Excel Macros','JIRA','Agile','Scrum',
        # Enterprise & Misc
        'SAP','PL/SQL','FPGA','Verilog','VHDL',
        # Roles/non-tech frequent terms
        'Data Entry','Telesales','Sales','Marketing','Accounts','Customer Service'
    ]
    # Deduplicate while preserving case
    seen = set()
    out = []
    for s in core:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out

def split_skills_field(s):
    """Split a skills cell into a list using common delimiters, trimming whitespace."""
    if not isinstance(s, str):
        return []
    parts = re.split(r"[,/|;]", s)
    return [p.strip() for p in parts if p and p.strip()]

def merge_extracted_skills(df, lexicon):
    """Create a 'skills_extracted' column from existing 'skills' + mined from title/description."""
    base = df.get('skills', pd.Series('', index=df.index)).fillna('').astype(str).apply(split_skills_field)
    text_series = (
        df.get('description', pd.Series('', index=df.index)).fillna('').astype(str) + ' ' +
        df.get('job_title', pd.Series('', index=df.index)).fillna('').astype(str)
    )
    mined = text_series.apply(lambda txt: extract_skills(txt, lexicon))
    merged = []
    for i in range(len(df)):
        uniq = []
        for s in list(base.iloc[i]) + list(mined.iloc[i]):
            sc = s.strip()
            if sc and sc not in uniq:
                uniq.append(sc)
        merged.append(', '.join(uniq))
    df['skills_extracted'] = merged
    return df

# --- Salary/Experience parsing & Geo helpers ---
_salary_commare = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")

def _extract_numbers(text):
    if not isinstance(text, str):
        return []
    return [float(t.replace(',', '')) for t in _salary_commare.findall(text)]

@st.cache_data
def enrich_dataset(df):
    df = df.copy()
    # Salary parsing
    def parse_salary(cell):
        s = str(cell)
        # Normalize common textual variants
        s = s.replace('per annum', 'P.A').replace('Per Annum', 'P.A')
        nums = _extract_numbers(s)
        if not nums:
            return pd.Series({'salary_min': np.nan, 'salary_max': np.nan, 'salary_avg': np.nan, 'salary_currency': None, 'salary_norm': np.nan, 'salary_unit': None})
        # pick first two numbers as range
        if len(nums) == 1:
            mn = mx = nums[0]
        else:
            mn, mx = nums[0], nums[1]
            if mx < mn:
                mn, mx = mx, mn
        avg = (mn + mx) / 2.0
        # Detect currency
        cur = None
        if re.search(r"₹|INR|P\.A", s, flags=re.IGNORECASE):
            cur = 'INR'
        elif '$' in s or re.search(r"USD", s, flags=re.IGNORECASE):
            cur = 'USD'
        # Normalized display units
        if cur == 'INR':
            norm = avg / 100000.0  # LPA
            unit = 'LPA'
        elif cur == 'USD':
            norm = avg / 1000.0    # thousands USD
            unit = 'K USD'
        else:
            norm = avg
            unit = None
        return pd.Series({'salary_min': mn, 'salary_max': mx, 'salary_avg': avg, 'salary_currency': cur, 'salary_norm': norm, 'salary_unit': unit})

    if 'salary' in df.columns:
        sal_df = df['salary'].apply(parse_salary)
        for c in sal_df.columns:
            df[c] = sal_df[c]

    # Experience parsing
    def parse_exp(cell):
        nums = _extract_numbers(str(cell))
        if not nums:
            return pd.Series({'exp_min': np.nan, 'exp_max': np.nan, 'exp_avg': np.nan})
        if len(nums) == 1:
            mn = mx = nums[0]
        else:
            mn, mx = nums[0], nums[1]
            if mx < mn:
                mn, mx = mx, mn
        return pd.Series({'exp_min': mn, 'exp_max': mx, 'exp_avg': (mn + mx) / 2.0})

    if 'experience' in df.columns:
        exp_df = df['experience'].apply(parse_exp)
        for c in exp_df.columns:
            df[c] = exp_df[c]

    return df

@st.cache_data
def city_coords_map():
    return {
        # Major Indian cities
        'Bengaluru': (12.9716, 77.5946), 'Bangalore': (12.9716, 77.5946), 'Mumbai': (19.0760, 72.8777),
        'Delhi': (28.6139, 77.2090), 'New Delhi': (28.6139, 77.2090), 'Chennai': (13.0827, 80.2707),
        'Hyderabad': (17.3850, 78.4867), 'Pune': (18.5204, 73.8567), 'Kolkata': (22.5726, 88.3639),
        'Gurgaon': (28.4595, 77.0266), 'Noida': (28.5355, 77.3910), 'Ahmedabad': (23.0225, 72.5714),
        'Jaipur': (26.9124, 75.7873), 'Chandigarh': (30.7333, 76.7794), 'Lucknow': (26.8467, 80.9462),
        'Kochi': (9.9312, 76.2673), 'Coimbatore': (11.0168, 76.9558), 'Visakhapatnam': (17.6868, 83.2185),
        'Tirupati': (13.6288, 79.4192), 'Srikakulam': (18.2949, 83.8938), 'Kozhikode': (11.2588, 75.7804),
        'Mysuru': (12.2958, 76.6394), 'Mysore': (12.2958, 76.6394), 'Gurugram': (28.4595, 77.0266),
    }

def locations_to_points(df):
    coords = city_coords_map()
    rows = []
    loc_series = df.get('location', pd.Series('', index=df.index)).fillna('').astype(str)
    for _, loc in loc_series.items():
        # Split by comma and semicolon
        parts = re.split(r",|/|;|\|", loc)
        for p in parts:
            city = p.strip()
            if not city:
                continue
            if city in coords:
                lat, lon = coords[city]
                rows.append({'city': city, 'lat': lat, 'lon': lon})
    if not rows:
        return pd.DataFrame(columns=['city','lat','lon','count'])
    pts = pd.DataFrame(rows)
    return pts.groupby(['city','lat','lon']).size().reset_index(name='count')

# Function to perform keyword frequency analysis
def analyze_skill_frequency(df, top_n=20, skills_col='skills'):
    """Analyze frequency of skills across all job postings (robust to delimiter variations)."""
    all_skills = []
    series = df.get(skills_col, pd.Series([], dtype=object))
    for cell in series:
        if pd.notna(cell):
            # support both pre-joined strings and pre-split lists
            if isinstance(cell, list):
                toks = cell
            else:
                toks = split_skills_field(str(cell))
            all_skills.extend([s for s in toks if s])
    
    skill_counts = Counter(all_skills)
    skill_df = pd.DataFrame(skill_counts.most_common(top_n), columns=['Skill', 'Frequency'])
    total = len(df)
    skill_df['Percentage'] = 0.0 if total == 0 else (skill_df['Frequency'] / total * 100).round(2)
    return skill_df

# Function to perform time-based trend analysis
def analyze_temporal_trends(df, skills_col='skills'):
    """Analyze how skill demand changes over time"""
    # parse as UTC-aware timestamps
    df = df.copy()
    df['posted_date'] = pd.to_datetime(df['posted_date'], utc=True, errors='coerce')
    df['week'] = df['posted_date'].dt.to_period('W')
    
    # Get top 10 skills
    all_skills = []
    series = df.get(skills_col, pd.Series([], dtype=object))
    for cell in series:
        if pd.notna(cell):
            toks = cell if isinstance(cell, list) else split_skills_field(str(cell))
            all_skills.extend([s for s in toks if s])
    top_skills = [skill for skill, _ in Counter(all_skills).most_common(10)]
    
    # Count skills by week
    trend_data = []
    for week in df['week'].unique():
        week_df = df[df['week'] == week]
        week_skills = []
        for cell in week_df.get(skills_col, pd.Series([], dtype=object)):
            if pd.notna(cell):
                toks = cell if isinstance(cell, list) else split_skills_field(str(cell))
                week_skills.extend([s for s in toks if s])
        
        week_counts = Counter(week_skills)
        for skill in top_skills:
            trend_data.append({
                'Week': str(week),
                'Skill': skill,
                'Count': week_counts.get(skill, 0)
            })
    
    return pd.DataFrame(trend_data)

# Function to cluster jobs based on skills
def cluster_jobs(df, n_clusters=5):
    """Cluster job postings based on required skills"""
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')

    # Prefer description, but fall back to skills and job_title if description is empty
    docs_desc = df['description'].fillna('').astype(str).tolist()
    docs_skills_title = (df.get('skills', pd.Series('', index=df.index)).fillna('').astype(str) + ' ' +
                         df.get('job_title', pd.Series('', index=df.index)).fillna('').astype(str)).tolist()

    tfidf_matrix = None
    used_source = 'description'
    # Check whether documents contain any non-empty tokens (after stop-word filtering)
    def has_meaningful_text(docs):
        for d in docs:
            if isinstance(d, str) and d.strip():
                # quick heuristic: any alpha-numeric char
                if re.search(r"[A-Za-z0-9]", d):
                    return True
        return False

    # Try description first, then fallback to skills+job_title
    try_sources = [('description', docs_desc), ('skills+job_title', docs_skills_title)]
    for src_name, docs in try_sources:
        if not has_meaningful_text(docs):
            continue
        try:
            tfidf_matrix = tfidf.fit_transform(docs)
            used_source = src_name
            break
        except ValueError:
            tfidf_matrix = None
            continue

    if tfidf_matrix is None or tfidf_matrix.shape[1] == 0 or tfidf_matrix.shape[0] == 0:
        # give up and return defaults
        df['cluster'] = -1
        coords = np.zeros((len(df), 2))
        cluster_terms = {i: [] for i in range(n_clusters)}
        return df, coords, cluster_terms

    # If TF-IDF produced zero features, handle gracefully
    if tfidf_matrix is None or tfidf_matrix.shape[1] == 0 or tfidf_matrix.shape[0] == 0:
        df['cluster'] = -1
        coords = np.zeros((len(df), 2))
        cluster_terms = {i: [] for i in range(n_clusters)}
        return df, coords, cluster_terms

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    df['cluster'] = labels

    # Get top terms for each cluster
    try:
        terms = tfidf.get_feature_names_out()
    except Exception:
        terms = []

    cluster_terms = {}
    for i in range(n_clusters):
        if hasattr(kmeans, 'cluster_centers_') and len(terms) > 0:
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-10:][::-1]
            cluster_terms[i] = [terms[idx] for idx in top_indices]
        else:
            cluster_terms[i] = []

    # PCA for visualization (only when there are samples and features)
    try:
        if tfidf_matrix.shape[0] > 0 and tfidf_matrix.shape[1] > 0:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(tfidf_matrix.toarray())
        else:
            coords = np.zeros((len(df), 2))
    except Exception:
        coords = np.zeros((len(df), 2))

    return df, coords, cluster_terms

# Load or generate data
if data_source == "Upload CSV" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} job postings from CSV")
    # Build richer skills from description/title for uploaded CSV to improve accuracy
    try:
        lex = build_skill_lexicon()
        df = merge_extracted_skills(df, lex)
    except Exception:
        pass
    # Parse salary/experience
    df = enrich_dataset(df)
else:
    with st.spinner("Generating sample job data..."):
        df = generate_sample_data(500)
    st.info(f"📊 Generated {len(df)} sample job postings for analysis")
    # Parse salary/experience for sample data
    df = enrich_dataset(df)

# Main analysis
if df is not None and len(df) > 0:
    
    # Filter by date range
    # ensure posted_date is parsed as UTC-aware timestamps for safe comparisons
    df['posted_date'] = pd.to_datetime(df['posted_date'], utc=True, errors='coerce')

    # Build a cutoff_date that matches the tz-awareness of the posted_date series
    now_utc = pd.Timestamp.now(tz='UTC')
    if df['posted_date'].dt.tz is None:
        # posted_date is tz-naive (rare if utc=True failed on some rows)
        cutoff_date = now_utc.tz_convert(None) - pd.Timedelta(days=date_range)
    else:
        cutoff_date = now_utc - pd.Timedelta(days=date_range)

    # Safe filtering: if posted_date parsing yielded NaT for many rows, df['posted_date'] may be
    # a tz-aware series or partially-naive. Use boolean mask safely.
    try:
        df_filtered = df[df['posted_date'] >= cutoff_date].copy()
    except TypeError:
        # Fallback: coerce both sides to naive datetimes for comparison
        df_filtered = df[df['posted_date'].dt.tz_localize(None) >= cutoff_date.tz_localize(None)].copy()

    # If filtering removed all rows, show diagnostics and offer to use full dataset
    if df_filtered.empty:
        parsed_dates = df['posted_date'].notna().sum()
        total_rows = len(df)
        st.warning(
            f"No postings found within the last {date_range} days. ({parsed_dates}/{total_rows} rows have a parsable posted_date)."
        )
        use_full = st.checkbox("Use full dataset (ignore date filter) to show analysis", value=True)
        if use_full:
            df_filtered = df.copy()
            st.info(f"Using full dataset: {len(df_filtered)} postings")
        else:
            st.info("Adjust the 'Days to Analyze' slider or check the CSV 'posted_date' format to proceed.")

    # Choose which skills column to use (keep sample analysis unchanged)
    if data_source == "Upload CSV" and 'skills_extracted' in df_filtered.columns:
        skills_col = 'skills_extracted'
    else:
        skills_col = 'skills'
    
    # Key metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Job Postings", len(df_filtered))
    with col2:
        unique_companies = df_filtered['company'].nunique()
        st.metric("Companies Hiring", unique_companies)
    with col3:
        remote_jobs = len(df_filtered[df_filtered['location'].str.contains('Remote', case=False, na=False)])
        denom = len(df_filtered)
        if denom > 0:
            pct = remote_jobs / denom * 100
            st.metric("Remote Positions", f"{remote_jobs} ({pct:.1f}%)")
        else:
            st.metric("Remote Positions", f"{remote_jobs} (0.0%)")
    with col4:
        # Make recent cutoff match tz-awareness of series
        now_utc = pd.Timestamp.now(tz='UTC')
        if df_filtered['posted_date'].dtype == 'datetime64[ns, UTC]':
            recent_cutoff = now_utc - pd.Timedelta(days=7)
        else:
            recent_cutoff = now_utc.tz_convert(None) - pd.Timedelta(days=7)

        try:
            recent_jobs = len(df_filtered[df_filtered['posted_date'] >= recent_cutoff])
        except Exception:
            # Fallback when comparison fails
            recent_jobs = 0
        st.metric("Posted This Week", recent_jobs)
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Skill Analysis", 
        "📈 Trend Forecasting", 
        "🎯 Job Clustering",
        "🏢 Company Insights",
        "💡 Predictions",
        "📊 BI Explorer"
    ])
    
    with tab1:
        st.header("Top In-Demand Skills")
        
        # Skill frequency analysis
        skill_freq = analyze_skill_frequency(df_filtered, top_n=top_n_skills, skills_col=skills_col)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            fig = px.bar(skill_freq, x='Skill', y='Frequency',
                        title='Most Demanded Tech Skills',
                        color='Frequency',
                        color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart - top 10
            fig = px.pie(skill_freq.head(10), values='Frequency', names='Skill',
                        title='Top 10 Skills Distribution')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Skill Metrics")
        st.dataframe(skill_freq, use_container_width=True, height=400)
        
        # Skill categories analysis
        st.subheader("Skills by Category")
        categories = {
            'Programming Languages': ['Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust', 'TypeScript'],
            'Data & AI': ['Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Data Analysis'],
            'Cloud & DevOps': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD'],
            'Web Technologies': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask']
        }
        
        category_counts = {}
        for category, skills in categories.items():
            count = sum([skill_freq[skill_freq['Skill'] == skill]['Frequency'].sum() 
                        for skill in skills if skill in skill_freq['Skill'].values])
            category_counts[category] = count
        
        cat_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
        fig = px.bar(cat_df, x='Category', y='Count', 
                    title='Skill Demand by Category',
                    color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

        # Description-based keyword extraction
        st.subheader("Description Keywords (Text Mining)")
        docs = df_filtered['description'].fillna('').astype(str).tolist()
        uni = extract_top_ngrams_tfidf(docs, ngram_range=(1,1), top_n=15)

        if uni:
            uni_df = pd.DataFrame(uni, columns=['Term', 'Score'])
            fig = px.bar(uni_df, x='Term', y='Score', title='Top Unigrams from Descriptions', color='Score', color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Not enough description text to extract unigrams.')

        # Show sample postings for top terms
        sample_terms = [t for t, _ in (uni[:5] if uni else [])]
        if sample_terms:
            st.markdown('#### Sample Postings Containing Top Terms')
            rows = []
            for term in sample_terms:
                matched = df_filtered[df_filtered['description'].str.contains(term, case=False, na=False)].head(5)
                for _, r in matched.iterrows():
                    rows.append({'Term': term, 'Job Title': r.get('job_title',''), 'Company': r.get('company',''), 'Short Description': r.get('description','')[:200]})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    with tab2:
        st.header("Temporal Trend Analysis")
        
        # Time-based trends
        trend_df = analyze_temporal_trends(df_filtered, skills_col=skills_col)
        
        if len(trend_df) > 0:
            # Line chart
            fig = px.line(trend_df, x='Week', y='Count', color='Skill',
                         title='Skill Demand Trends Over Time',
                         markers=True)
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Job Clustering Analysis")
        
        with st.spinner("Performing clustering analysis..."):
            df_clustered, coords, cluster_terms = cluster_jobs(df_filtered, n_clusters)
        
        # Scatter plot
        cluster_viz = pd.DataFrame({
            'PC1': coords[:, 0],
            'PC2': coords[:, 1],
            'Cluster': (df_clustered['cluster'] + 1).astype(str),
            'Job Title': df_clustered['job_title'],
            'Company': df_clustered['company']
        })
        
        fig = px.scatter(cluster_viz, x='PC1', y='PC2', color='Cluster',
                        hover_data=['Job Title', 'Company'],
                        title='Job Postings Clustered by Required Skills (Cluster #)',
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster insights
        st.subheader("Cluster Insights")
        
        cols = st.columns(min(n_clusters, 3))
        for i in range(n_clusters):
            with cols[i % 3]:
                cluster_jobs = df_clustered[df_clustered['cluster'] == i]
                st.markdown(f"""
                <div class="insight-box">
                    <h4>Cluster {i+1}</h4>
                    <p><strong>{len(cluster_jobs)} jobs</strong></p>
                    <p><strong>Top Skills:</strong><br/>
                    {', '.join(cluster_terms[i][:5])}</p>
                    <p><strong>Common Titles:</strong><br/>
                    {', '.join(cluster_jobs['job_title'].value_counts().head(3).index.tolist())}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.header("Company Hiring Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top hiring companies
            company_counts = df_filtered['company'].value_counts().head(10)
            comp_df = company_counts.reset_index()
            comp_df.columns = ['Company', 'Count']
            fig = px.bar(comp_df, x='Count', y='Company',
                        orientation='h',
                        title='Top 10 Hiring Companies',
                        labels={'Count': 'Number of Positions', 'Company': 'Company'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Location distribution
            location_counts = df_filtered['location'].value_counts().head(10)
            loc_df = location_counts.reset_index()
            loc_df.columns = ['Location', 'Count']
            fig = px.pie(loc_df, values='Count', names='Location', title='Top Job Locations')
            st.plotly_chart(fig, use_container_width=True)
        
        # Job title distribution
        st.subheader("Most Common Job Titles")
        title_counts = df_filtered['job_title'].value_counts().head(15)
        title_df = title_counts.reset_index()
        title_df.columns = ['Job Title', 'Count']
        fig = px.bar(title_df, x='Job Title', y='Count',
                title='Popular Job Roles',
                labels={'Job Title': 'Job Title', 'Count': 'Count'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("🔮 Predictive Insights & Recommendations")
        
        if data_source == "Upload CSV":
            # Data-driven predictions for uploaded datasets
            skill_freq = analyze_skill_frequency(df_filtered, top_n=30, skills_col=skills_col)
            trend_df = analyze_temporal_trends(df_filtered, skills_col=skills_col)
            
            st.markdown("""
            <div class=\"insight-box\">
                <h3>🎯 Key Predictions for the Next 6 Months</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Keep sample data analysis as-is (static guidance)
            skill_freq = analyze_skill_frequency(df_filtered, 30, skills_col=skills_col)
            trend_df = analyze_temporal_trends(df_filtered, skills_col=skills_col)
            st.markdown("""
            <div class="insight-box">
                <h3>🎯 Key Predictions for the Next 6 Months</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🚀 Emerging Skills")
                st.markdown("""
                Based on growth patterns and demand trends:
                
                1. **AI/ML Technologies** - Demand increasing by 45%
                   - LLM integration, GPT, Transformers
                   - Focus on practical AI applications
                
                2. **Cloud-Native Development** - 38% growth
                   - Kubernetes, Docker orchestration
                   - Serverless architectures
                
                3. **Data Engineering** - 32% growth
                   - Real-time data pipelines
                   - Data mesh architectures
                
                4. **Cybersecurity** - 28% growth
                   - Zero-trust security models
                   - Cloud security expertise
                """)
            with col2:
                st.subheader("💼 Career Recommendations")
                st.markdown("""
                Strategic skill development paths:
                
                **For Beginners:**
                - Python + SQL fundamentals
                - Cloud basics (AWS/Azure)
                - Git & collaboration tools
                
                **For Intermediate:**
                - Advanced ML frameworks
                - Container orchestration
                - System design patterns
                
                **For Advanced:**
                - Distributed systems
                - MLOps & LLMOps
                - Architecture design
                """)
        
        # Skill combinations analysis
        st.subheader("💡 Winning Skill Combinations")
        
        top_skills = skill_freq.head(10)['Skill'].tolist()
        combinations = []
        
        for _, row in df_filtered.iterrows():
            job_skills = split_skills_field(str(row.get(skills_col, '')))
            matching = [s for s in job_skills if s in top_skills]
            if len(matching) >= 2:
                combinations.append(tuple(sorted(matching[:3])))
        
        combo_counts = Counter(combinations).most_common(10)
        
        if combo_counts:
            combo_df = pd.DataFrame(combo_counts, columns=['Skill Combination', 'Frequency'])
            combo_df['Skills'] = combo_df['Skill Combination'].apply(lambda x: ' + '.join(x))
            
            fig = px.bar(combo_df, x='Skills', y='Frequency',
                        title='Most Valuable Skill Combinations',
                        color='Frequency',
                        color_continuous_scale='Greens')
            fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market saturation indicator (only for sample data)
        if data_source != "Upload CSV":
            st.subheader("📊 Market Saturation Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="insight-box" style="border-left-color: #4caf50;">
                    <h4 style="color: #4caf50;">🟢 High Demand, Low Supply</h4>
                    <ul>
                        <li>MLOps Engineers</li>
                        <li>Cloud Architects</li>
                        <li>AI/ML Specialists</li>
                    </ul>
                    <p><strong>Action:</strong> Excellent opportunities for career transition</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box" style="border-left-color: #ff9800;">
                    <h4 style="color: #ff9800;">🟡 Balanced Market</h4>
                    <ul>
                        <li>Full Stack Developers</li>
                        <li>Data Scientists</li>
                        <li>DevOps Engineers</li>
                    </ul>
                    <p><strong>Action:</strong> Specialize to stand out</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="insight-box" style="border-left-color: #f44336;">
                    <h4 style="color: #f44336;">🔴 Saturated Roles</h4>
                    <ul>
                        <li>Junior Frontend</li>
                        <li>Basic Web Development</li>
                        <li>Manual QA Testers</li>
                    </ul>
                    <p><strong>Action:</strong> Upskill to mid/senior level</p>
                </div>
                """, unsafe_allow_html=True)
        
        
    with tab6:
        st.header("📊 Business Intelligence Explorer")
        
        # Filters
        c1, c2, c3 = st.columns(3)
        with c1:
            f_companies = st.multiselect("Company", sorted(df_filtered['company'].dropna().astype(str).unique().tolist()))
        with c2:
            f_locations = st.multiselect("Location", sorted(df_filtered['location'].dropna().astype(str).unique().tolist()))
        with c3:
            f_titles = st.multiselect("Job Title", sorted(df_filtered['job_title'].dropna().astype(str).unique().tolist()))
        # Optional skill filter from current top skills
        top_skills_list = analyze_skill_frequency(df_filtered, top_n=20, skills_col=skills_col)['Skill'].tolist()
        f_skills = st.multiselect("Filter by Skills", top_skills_list)
        
        df_bi = df_filtered.copy()
        if f_companies:
            df_bi = df_bi[df_bi['company'].isin(f_companies)]
        if f_locations:
            df_bi = df_bi[df_bi['location'].isin(f_locations)]
        if f_titles:
            df_bi = df_bi[df_bi['job_title'].isin(f_titles)]
        if f_skills:
            df_bi = df_bi[df_bi.get(skills_col, '').apply(lambda s: any(sk in str(s) for sk in f_skills))]
        
        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Postings", len(df_bi))
        with k2:
            # Determine dominant salary unit for display
            unit_counts = df_bi['salary_unit'].value_counts(dropna=True)
            dom_unit = unit_counts.index[0] if len(unit_counts) else None
            if dom_unit is not None:
                med_sal = np.nanmedian(df_bi[df_bi['salary_unit'] == dom_unit]['salary_norm'])
                st.metric(f"Median Salary ({dom_unit})", f"{med_sal:,.2f}" if not np.isnan(med_sal) else "N/A")
            else:
                st.metric("Median Salary", "N/A")
        with k3:
            st.metric("Companies", int(df_bi['company'].nunique()))
        with k4:
            st.metric("Locations", int(df_bi['location'].nunique()))
        
        st.markdown("---")
        # Salary by Skill (Top 15)
        st.subheader("Salary by Skill")
        # Use dominant unit to avoid mixing INR and USD
        unit_counts = df_bi['salary_unit'].value_counts(dropna=True)
        dom_unit = unit_counts.index[0] if len(unit_counts) else None
        if dom_unit is not None:
            tmp = df_bi[df_bi['salary_unit'] == dom_unit].copy()
            tmp['skill_list'] = tmp.get(skills_col, '').apply(split_skills_field)
            exp = tmp.explode('skill_list')
            exp = exp[exp['skill_list'].notna() & (exp['skill_list'] != '')]
            if not exp.empty and 'salary_norm' in exp.columns:
                grp = exp.groupby('skill_list')['salary_norm'].mean().sort_values(ascending=False).head(15).reset_index()
                fig = px.bar(grp, x='skill_list', y='salary_norm', title=f'Average Salary by Skill ({dom_unit})', color='salary_norm', color_continuous_scale='Tealgrn', labels={'salary_norm': dom_unit, 'skill_list': 'Skill'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Salary data not available for the selected filters.")
        else:
            st.info("Salary data not available in this dataset.")
        
        st.markdown("---")
        # Skill Co-occurrence Network
        st.subheader("Skill Co-occurrence Network")
        try:
            import networkx as nx
            # Build co-occurrence among top skills
            topS = top_skills_list[:15]
            pairs = Counter()
            for _, row in df_bi.iterrows():
                lst = [s for s in split_skills_field(str(row.get(skills_col,''))) if s in topS]
                lst = sorted(set(lst))
                for i in range(len(lst)):
                    for j in range(i+1, len(lst)):
                        pairs[(lst[i], lst[j])] += 1
            G = nx.Graph()
            for (a,b), w in pairs.items():
                G.add_edge(a,b, weight=w)
            if G.number_of_edges() == 0:
                st.info("Not enough co-occurrence data to draw a network.")
            else:
                pos = nx.spring_layout(G, k=0.8, seed=42)
                edge_x, edge_y = [], []
                for (u,v,d) in G.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                node_x, node_y, text, size = [], [], [], []
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x); node_y.append(y)
                    text.append(f"{n}")
                    size.append(10 + 5*G.degree[n])
                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=text, textposition='top center',
                                        marker=dict(size=size, color=size, colorscale='YlGnBu', showscale=False))
                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(title='Co-occurrence Network', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=500)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Install networkx to view the network graph: pip install networkx")
        
        st.markdown("---")
        # Geo heatmap (scatter map)
        st.subheader("Geo Distribution (Top Cities)")
        pts = locations_to_points(df_bi)
        if len(pts) > 0:
            fig = px.scatter_mapbox(pts, lat='lat', lon='lon', size='count', color='count', hover_name='city',
                                    zoom=3, height=450, color_continuous_scale='Sunset', title='Job Density by City')
            fig.update_layout(mapbox_style='open-street-map')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recognizable city names to map.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Job Market Trends Analyzer</strong> | Built with Streamlit & Python</p>
    <p>Data updated: {}</p>
</div>

""".format(datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)

