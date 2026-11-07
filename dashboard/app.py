from __future__ import annotations

from pathlib import Path
import re
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------
# StockSense 3D dashboard starter
# Place this file at <project-root>/dashboard/app.py.
# It searches the existing data/processed and data/raw folders.
# It never generates fake business metrics.
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="StockSense | Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

DARK = {
    "paper": "#07111f",
    "panel": "rgba(13, 28, 48, .82)",
    "line": "rgba(78, 190, 255, .20)",
    "cyan": "#43d9ff",
    "blue": "#4c8dff",
    "purple": "#a78bfa",
    "teal": "#2dd4bf",
    "text": "#e8f2ff",
    "muted": "#91a7c2",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
    .stApp {{
      background:
        radial-gradient(circle at 12% 4%, rgba(31,117,190,.20), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(110,65,190,.15), transparent 25%),
        linear-gradient(145deg, #050b14 0%, #081525 52%, #07101c 100%);
      color: {DARK['text']};
    }}
    /* Premium sidebar shell */
    [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #07111f 0%, #0b1930 55%, #101d35 100%);
      border-right: 1px solid rgba(67,217,255,.18);
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding: 1.35rem 1rem 1.5rem; }}
    .sidebar-brand {{ padding: 10px 9px 22px; }}
    .sidebar-logo {{ color:#f3f8ff; font-size:1.55rem; font-weight:800; letter-spacing:-.7px; }}
    .sidebar-subtitle {{ color:#7f9bb9; font-size:10px; letter-spacing:2px; margin-top:7px; font-weight:700; }}
    .sidebar-section {{ color:#6f91b5; font-size:10px; font-weight:800; letter-spacing:2px; margin:17px 0 10px 8px; }}
    [data-testid="stSidebar"] div[role="radiogroup"] {{ gap:6px; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
      background:rgba(255,255,255,.025); border:1px solid transparent;
      border-radius:10px; padding:8px 10px; transition:all .2s ease;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
      background:rgba(67,217,255,.09); border-color:rgba(67,217,255,.2);
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
      background:linear-gradient(100deg,rgba(25,143,190,.30),rgba(83,91,190,.23));
      border:1px solid rgba(67,217,255,.48); box-shadow:0 0 16px rgba(67,217,255,.10);
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label p {{ color:#dceaff; font-size:12.5px; font-weight:600; }}
    [data-testid="stSidebar"] .stButton button {{
      background:linear-gradient(100deg,#087da8,#424fba); border:1px solid rgba(67,217,255,.4);
      border-radius:11px; color:white; font-weight:650; padding:10px;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
      border-color:#43d9ff; box-shadow:0 0 18px rgba(67,217,255,.2);
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }}
    .brand-kicker {{
      color: {DARK['cyan']}; letter-spacing: .22em; text-transform: uppercase;
      font-size: .72rem; font-weight: 700;
    }}
    .hero-title {{
      font-size: clamp(2rem, 4vw, 3.7rem); line-height: 1.05;
      font-weight: 800; color: #f4f9ff; margin: .3rem 0 .55rem 0;
    }}
    .hero-subtitle {{ color: #a8bdd5; font-size: 1rem; max-width: 680px; }}
    .glass {{
      background: linear-gradient(135deg, rgba(18,40,66,.88), rgba(9,23,40,.72));
      border: 1px solid rgba(87,190,255,.20);
      border-radius: 18px; padding: 17px 19px;
      box-shadow: 0 12px 35px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.04);
      min-height: 112px;
    }}
    .glass:hover {{
      border-color: rgba(67,217,255,.55);
      box-shadow: 0 14px 38px rgba(0,0,0,.27), 0 0 22px rgba(67,217,255,.08);
      transform: translateY(-2px); transition: all .22s ease;
    }}
    .metric-label {{ color: #9bb1ca; font-size: .78rem; font-weight: 600; }}
    .metric-value {{ color: #f2f8ff; font-size: 1.75rem; font-weight: 800; margin-top: 8px; }}
    .metric-note {{ color: #6f8ba9; font-size: .72rem; margin-top: 3px; }}
    .section-title {{ font-size: 1.15rem; font-weight: 700; color: #eaf5ff; margin: .5rem 0 .2rem; }}
    .section-note {{ color: #8fa8c4; font-size: .82rem; margin-bottom: .7rem; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid rgba(87,190,255,.16); border-radius: 12px; }}
    .stButton > button, .stDownloadButton > button {{
      border: 1px solid rgba(67,217,255,.35); border-radius: 10px;
      background: linear-gradient(100deg, rgba(22,88,130,.75), rgba(47,65,133,.75));
      color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())

@st.cache_data(show_spinner=False)
def discover_csvs(root_str: str):
    root = Path(root_str)
    found = []
    for folder_name in ("data/processed", "data/raw", "reports"):
        folder = root / folder_name
        if folder.exists():
            found.extend(folder.rglob("*.csv"))
    # Avoid scanning virtual environments and keep paths deterministic.
    unique = sorted({p.resolve() for p in found if p.is_file()}, key=lambda p: (
        0 if "processed" in str(p).lower() else 1, str(p).lower()
    ))
    return [str(p) for p in unique]

@st.cache_data(show_spinner="Loading project datasets...")
def read_csv_safely(path: str):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1", low_memory=False)
        except Exception:
            return pd.DataFrame()

def choose_dataset(paths, keywords):
    candidates = []
    for path in paths:
        name = Path(path).name.lower()
        if any(k in name for k in keywords):
            candidates.append(path)
    # Prefer processed artifacts when there are multiple candidates.
    candidates.sort(key=lambda p: (0 if "processed" in p.lower() else 1, len(p)))
    for path in candidates:
        df = read_csv_safely(path)
        if not df.empty:
            return path, df
    return None, pd.DataFrame()

def find_col(df, candidates):
    if df.empty:
        return None
    lookup = {norm(c): c for c in df.columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    # Conservative partial matching for common schema variations.
    for candidate in candidates:
        key = norm(candidate)
        for col in df.columns:
            if key and (key in norm(col) or norm(col) in key):
                return col
    return None

def fmt_number(value, decimals=0):
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"

def metric_card(label, value, note="Calculated from available project data"):
    st.markdown(
        f"""<div class="glass">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-note">{html.escape(note)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def plotly_dark(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dcecff", family="Inter"),
        margin=dict(l=12, r=12, t=45, b=12),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(gridcolor="rgba(145,167,194,.10)", zerolinecolor="rgba(145,167,194,.10)")
    fig.update_yaxes(gridcolor="rgba(145,167,194,.10)", zerolinecolor="rgba(145,167,194,.10)")
    return fig

def render_3d_hero():
    # A lightweight Three.js scene. If the CDN is unavailable, the rest of
    # the dashboard still works normally.
    html_scene = r"""
    <div id="stocksense-3d" style="height:330px;width:100%;border-radius:18px;overflow:hidden;
      background:radial-gradient(circle at 50% 45%,#12345a 0%,#081626 58%,#07111e 100%);
      border:1px solid rgba(67,217,255,.22);position:relative;">
      <div style="position:absolute;z-index:2;left:18px;top:15px;color:#9edfff;font:600 11px Inter,sans-serif;letter-spacing:2px;">
        LIVE SUPPLY NETWORK · DIGITAL TWIN
      </div>
      <div id="three-fallback" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
        color:#8fb8d8;font:13px Inter,sans-serif;">Loading 3D supply-chain visualization…</div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
      const host=document.getElementById('stocksense-3d');
      const fallback=document.getElementById('three-fallback');
      if(!window.THREE){fallback.textContent='3D preview unavailable — dashboard analytics remain available.';return;}
      const scene=new THREE.Scene();
      const camera=new THREE.PerspectiveCamera(42,host.clientWidth/host.clientHeight,.1,100);
      camera.position.set(0,4.6,10.5);
      const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.7));
      renderer.setSize(host.clientWidth,host.clientHeight);
      host.appendChild(renderer.domElement);
      fallback.style.display='none';
      scene.add(new THREE.AmbientLight(0x9ccfff,1.0));
      const key=new THREE.PointLight(0x39cfff,2.2,30);key.position.set(2,6,5);scene.add(key);
      const purple=new THREE.PointLight(0x8666ff,1.5,25);purple.position.set(-5,2,-3);scene.add(purple);
      const group=new THREE.Group();scene.add(group);
      const floor=new THREE.Mesh(new THREE.CircleGeometry(5.4,64),
        new THREE.MeshBasicMaterial({color:0x0a2943,transparent:true,opacity:.45}));
      floor.rotation.x=-Math.PI/2;floor.position.y=-1.45;group.add(floor);
      const grid=new THREE.GridHelper(11,22,0x1d86b9,0x16415d);grid.position.y=-1.42;group.add(grid);
      const boxMat=new THREE.MeshStandardMaterial({color:0x167fa9,metalness:.45,roughness:.32,emissive:0x062d49});
      const accentMat=new THREE.MeshStandardMaterial({color:0x7658d9,metalness:.35,roughness:.3,emissive:0x21104a});
      // Warehouse racks and cartons
      for(let rack=0;rack<3;rack++){
        const x=(rack-1)*2.7;
        for(let level=0;level<3;level++){
          const shelf=new THREE.Mesh(new THREE.BoxGeometry(2.0,.10,.65),
            new THREE.MeshStandardMaterial({color:0x2d5778,metalness:.5,roughness:.4}));
          shelf.position.set(x,-.95+level*.92,-.5);group.add(shelf);
          for(let b=0;b<3;b++){
            const carton=new THREE.Mesh(new THREE.BoxGeometry(.42,.42,.42),b===1?accentMat:boxMat);
            carton.position.set(x-.62+b*.62,-.68+level*.92,-.5);group.add(carton);
          }
        }
        const postGeo=new THREE.BoxGeometry(.09,2.85,.10);
        [-.95,.95].forEach(dx=>{const post=new THREE.Mesh(postGeo,boxMat);post.position.set(x+dx,-.05,-.5);group.add(post);});
      }
      // Glowing network nodes and links
      const nodes=[[-4,.2,1.3],[0,1.15,1.7],[4,.2,1.3],[-2,-.4,-2],[2,-.4,-2]];
      const nodeMat=new THREE.MeshStandardMaterial({color:0x42dcff,emissive:0x1686bd,emissiveIntensity:1.6});
      nodes.forEach((p,i)=>{
        const n=new THREE.Mesh(new THREE.SphereGeometry(.13,20,20),nodeMat);
        n.position.set(...p);group.add(n);
        const ring=new THREE.Mesh(new THREE.TorusGeometry(.23,.018,8,32),
          new THREE.MeshBasicMaterial({color:i%2?0xa78bfa:0x43d9ff}));
        ring.position.copy(n.position);ring.rotation.x=Math.PI/2;group.add(ring);
      });
      const links=[[0,1],[1,2],[0,3],[3,4],[4,2],[1,4]];
      links.forEach(pair=>{
        const a=new THREE.Vector3(...nodes[pair[0]]),b=new THREE.Vector3(...nodes[pair[1]]);
        const curve=new THREE.QuadraticBezierCurve3(a,new THREE.Vector3((a.x+b.x)/2,2.1,(a.z+b.z)/2),b);
        const line=new THREE.Mesh(new THREE.TubeGeometry(curve,36,.018,6,false),
          new THREE.MeshBasicMaterial({color:0x37cfff,transparent:true,opacity:.65}));
        group.add(line);
      });
      let dragging=false,lastX=0,lastY=0,rotY=0,rotX=0;
      renderer.domElement.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;});
      window.addEventListener('pointerup',()=>dragging=false);
      window.addEventListener('pointermove',e=>{
        if(!dragging)return;
        rotY+=(e.clientX-lastX)*.006;rotX+=(e.clientY-lastY)*.003;
        lastX=e.clientX;lastY=e.clientY;
      });
      renderer.domElement.addEventListener('wheel',e=>{
        camera.position.z=Math.max(7,Math.min(14,camera.position.z+e.deltaY*.006));
      },{passive:true});
      let tick=0;
      function animate(){
        requestAnimationFrame(animate);tick+=.008;
        group.rotation.y=rotY+Math.sin(tick*.35)*.10;
        group.rotation.x=Math.max(-.22,Math.min(.22,rotX+Math.sin(tick*.25)*.025));
        renderer.render(scene,camera);
      }
      animate();
      const resize=()=>{const w=host.clientWidth,h=host.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);};
      new ResizeObserver(resize).observe(host);
    })();
    </script>
    """
    components.html(html_scene, height=345, scrolling=False)

# ---------------------------------------------------------------------
# PREMIUM STOCKSENSE SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo">📦 StockSense</div>
            <div class="sidebar-subtitle">SUPPLY CHAIN INTELLIGENCE</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">WORKSPACE</div>', unsafe_allow_html=True)

    page_labels = {
        "Executive Overview": "🏠  Executive Overview",
        "Sales Intelligence": "📈  Sales Intelligence",
        "Inventory Intelligence": "📦  Inventory Intelligence",
        "AI Demand Forecasting": "🧠  AI Demand Forecasting",
        "Stockout Prediction": "⚠️  Stockout Prediction",
        "Inventory Optimization": "🎯  Inventory Optimization",
        "Procurement Intelligence": "🚚  Procurement Intelligence",
        "Model Performance": "📊  Model Performance",
        "About StockSense": "ℹ️  About StockSense",
    }
    page = st.radio(
        "Navigation",
        options=list(page_labels.keys()),
        format_func=lambda item: page_labels[item],
        label_visibility="collapsed",
        key="stocksense_navigation",
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">DATA CONNECTION</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background:rgba(45,212,191,.07);border:1px solid rgba(45,212,191,.18);
                    border-radius:10px;padding:12px;margin:4px 0 12px;">
            <div style="color:#2dd4bf;font-size:11px;font-weight:800;letter-spacing:1px;">
                ● LOCAL DATA WORKSPACE
            </div>
            <div style="color:#91a7c2;font-size:10px;margin-top:5px;">
                Project files are read from your local workspace
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("↻  Refresh Data Files", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
        <div style="margin-top:30px;padding-top:14px;border-top:1px solid rgba(120,160,200,.15);
                    color:#607d9c;font-size:9px;line-height:1.8;text-align:center;letter-spacing:1px;">
            STOCKSENSE AI PLATFORM<br>INVENTORY · FORECASTING · OPTIMIZATION
        </div>
    """, unsafe_allow_html=True)

csv_paths = discover_csvs(str(ROOT))
sales_path, sales = choose_dataset(csv_paths, ["sales", "sale", "transaction"])
inventory_path, inventory = choose_dataset(csv_paths, ["inventory", "stock"])
procurement_path, procurement = choose_dataset(csv_paths, ["procurement", "purchase", "invoice", "supplier"])

# Resolve likely columns using common variants from the StockSense source files.
sales_date = find_col(sales, ["Transaction Date", "date", "sales_date"])
sales_amount = find_col(sales, ["Sales Amount", "revenue", "amount", "sales"])
sales_qty = find_col(sales, ["Qty Sold", "quantity sold", "quantity", "units"])
sales_product = find_col(sales, ["Product No", "product_id", "sku", "product"])
sales_category = find_col(sales, ["Product Category", "category"])
sales_store = find_col(sales, ["Store", "store_id", "store name"])

inv_qty = find_col(inventory, ["Qty on hand", "quantity on hand", "stock quantity", "on_hand", "stock"])
inv_value = find_col(inventory, ["Cost of Stocks", "inventory value", "stock value", "Stocks Selling Amount"])
inv_product = find_col(inventory, ["Product No", "product_id", "sku", "product"])
inv_store = find_col(inventory, ["Store", "store_id", "store name"])
inv_status = find_col(inventory, ["Stock Status", "status"])

def safe_sum(df, col):
    if df.empty or not col:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    return values.sum(min_count=1)

def safe_nunique(df, col):
    return int(df[col].nunique(dropna=True)) if not df.empty and col else None

def chart_message(message):
    st.info(message)

def render_header(title, subtitle):
    st.markdown('<div class="brand-kicker">STOCKSENSE INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)
    st.write("")

def show_data_status():
    with st.expander("Inspect connected source files"):
        for label, path, df in [
            ("Sales", sales_path, sales),
            ("Inventory", inventory_path, inventory),
            ("Procurement", procurement_path, procurement),
        ]:
            if path:
                st.write(f"**{label}:** `{Path(path).relative_to(ROOT)}` — {len(df):,} rows × {len(df.columns)} columns")
                st.caption("Columns: " + ", ".join(map(str, df.columns)))
            else:
                st.warning(f"No {label.lower()} CSV was discovered in data/processed, data/raw, or reports.")

# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
if page == "Executive Overview":
    render_header("Supply chain, made intelligent.", "A unified view of retail performance, inventory health, and operational decisions.")
    left, right = st.columns([1.15, 1])
    with left:
        render_3d_hero()
    with right:
        st.markdown('<div class="section-title">Your operations at a glance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-note">Metrics are calculated from detected project files. Missing fields are shown as unavailable.</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            metric_card("Sales records", fmt_number(len(sales)) if not sales.empty else "—")
            metric_card("Products in sales", fmt_number(safe_nunique(sales, sales_product)))
        with b:
            metric_card("Inventory records", fmt_number(len(inventory)) if not inventory.empty else "—")
            metric_card("Stores in sales", fmt_number(safe_nunique(sales, sales_store)))
    st.write("")
    cols = st.columns(4)
    values = [
        ("Total sales value", fmt_number(safe_sum(sales, sales_amount), 2), "Sum of detected sales amount"),
        ("Units sold", fmt_number(safe_sum(sales, sales_qty)), "Sum of detected quantity"),
        ("Inventory quantity", fmt_number(safe_sum(inventory, inv_qty)), "Sum of detected on-hand quantity"),
        ("Inventory value", fmt_number(safe_sum(inventory, inv_value), 2), "Sum of detected inventory value"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col:
            metric_card(label, value, note)
    st.markdown('<div class="section-title">Performance snapshot</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if not sales.empty and sales_date and sales_amount:
            tmp = sales[[sales_date, sales_amount]].copy()
            tmp[sales_date] = pd.to_datetime(tmp[sales_date], errors="coerce")
            tmp[sales_amount] = pd.to_numeric(tmp[sales_amount], errors="coerce")
            tmp = tmp.dropna().groupby(pd.Grouper(key=sales_date, freq="MS"))[sales_amount].sum().reset_index()
            fig = px.area(tmp, x=sales_date, y=sales_amount, title="Monthly sales value")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)
        else:
            chart_message("Monthly sales chart needs a recognized transaction-date and sales-amount column.")
    with c2:
        if not inventory.empty and inv_qty:
            tmp = inventory.copy()
            tmp[inv_qty] = pd.to_numeric(tmp[inv_qty], errors="coerce")
            fig = px.histogram(tmp.dropna(subset=[inv_qty]), x=inv_qty, nbins=35, title="On-hand inventory distribution")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)
        else:
            chart_message("Inventory distribution needs a recognized on-hand quantity column.")
    show_data_status()

elif page == "Sales Intelligence":
    render_header("Sales intelligence", "Explore sales performance across time, products, categories, and stores.")
    if sales.empty:
        chart_message("No sales CSV found. Add your sales data under data/processed or data/raw.")
    else:
        filtered = sales.copy()
        f1, f2, f3 = st.columns(3)
        if sales_category:
            opts = ["All"] + sorted(filtered[sales_category].dropna().astype(str).unique().tolist())
            selected = f1.selectbox("Category", opts)
            if selected != "All":
                filtered = filtered[filtered[sales_category].astype(str) == selected]
        if sales_store:
            opts = ["All"] + sorted(filtered[sales_store].dropna().astype(str).unique().tolist())
            selected = f2.selectbox("Store", opts)
            if selected != "All":
                filtered = filtered[filtered[sales_store].astype(str) == selected]
        if sales_date:
            dates = pd.to_datetime(filtered[sales_date], errors="coerce").dropna()
            if not dates.empty:
                start, end = f3.date_input("Date range", (dates.min().date(), dates.max().date()))
                d = pd.to_datetime(filtered[sales_date], errors="coerce")
                filtered = filtered[(d.dt.date >= start) & (d.dt.date <= end)]
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("Filtered records", fmt_number(len(filtered)))
        with m2: metric_card("Sales value", fmt_number(safe_sum(filtered, sales_amount), 2))
        with m3: metric_card("Units sold", fmt_number(safe_sum(filtered, sales_qty)))
        c1, c2 = st.columns(2)
        with c1:
            if sales_date and sales_amount:
                tmp = filtered[[sales_date, sales_amount]].copy()
                tmp[sales_date] = pd.to_datetime(tmp[sales_date], errors="coerce")
                tmp[sales_amount] = pd.to_numeric(tmp[sales_amount], errors="coerce")
                tmp = tmp.dropna().groupby(pd.Grouper(key=sales_date, freq="W"))[sales_amount].sum().reset_index()
                st.plotly_chart(plotly_dark(px.line(tmp, x=sales_date, y=sales_amount, markers=True, title="Weekly sales value")), use_container_width=True)
            else: chart_message("Date and sales amount columns are required for the trend chart.")
        with c2:
            if sales_product and sales_qty:
                tmp = filtered.copy()
                tmp[sales_qty] = pd.to_numeric(tmp[sales_qty], errors="coerce")
                tmp = tmp.groupby(sales_product, dropna=True)[sales_qty].sum().nlargest(12).sort_values().reset_index()
                st.plotly_chart(plotly_dark(px.bar(tmp, x=sales_qty, y=sales_product, orientation="h", title="Top products by units sold")), use_container_width=True)
            else: chart_message("Product and quantity columns are required for product ranking.")
        with st.expander("View filtered sales data"):
            st.dataframe(filtered.head(1000), use_container_width=True)
            st.download_button("Download filtered sales CSV", filtered.to_csv(index=False).encode("utf-8"), "stocksense_sales_filtered.csv", "text/csv")

elif page == "Inventory Intelligence":
    render_header("Inventory intelligence", "Monitor stock levels, inventory value, and product-store availability.")
    if inventory.empty:
        chart_message("No inventory CSV found. Add inventory data under data/processed or data/raw.")
    else:
        f1, f2 = st.columns(2)
        filtered = inventory.copy()
        if inv_store:
            stores = ["All"] + sorted(filtered[inv_store].dropna().astype(str).unique().tolist())
            s = f1.selectbox("Store", stores)
            if s != "All": filtered = filtered[filtered[inv_store].astype(str) == s]
        if inv_status:
            statuses = ["All"] + sorted(filtered[inv_status].dropna().astype(str).unique().tolist())
            s = f2.selectbox("Stock status", statuses)
            if s != "All": filtered = filtered[filtered[inv_status].astype(str) == s]
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Inventory rows", fmt_number(len(filtered)))
        with c2: metric_card("On-hand units", fmt_number(safe_sum(filtered, inv_qty)))
        with c3: metric_card("Inventory value", fmt_number(safe_sum(filtered, inv_value), 2))
        a, b = st.columns(2)
        with a:
            if inv_qty:
                tmp = filtered.copy()
                tmp[inv_qty] = pd.to_numeric(tmp[inv_qty], errors="coerce")
                st.plotly_chart(plotly_dark(px.histogram(tmp.dropna(subset=[inv_qty]), x=inv_qty, nbins=35, title="Stock quantity distribution")), use_container_width=True)
            else: chart_message("No recognizable on-hand quantity column.")
        with b:
            if inv_status:
                tmp = filtered[inv_status].fillna("Unknown").astype(str).value_counts().rename_axis("Status").reset_index(name="Records")
                st.plotly_chart(plotly_dark(px.pie(tmp, names="Status", values="Records", hole=.58, title="Inventory status")), use_container_width=True)
            else: chart_message("No stock-status column found.")
        with st.expander("View inventory records"):
            st.dataframe(filtered.head(1000), use_container_width=True)
            st.download_button("Download inventory CSV", filtered.to_csv(index=False).encode("utf-8"), "stocksense_inventory.csv", "text/csv")

elif page == "AI Demand Forecasting":
    render_header("AI demand forecasting", "Connect this page to the forecasting artifacts generated by your training pipeline.")
    forecast_files = [p for p in csv_paths if any(k in Path(p).name.lower() for k in ["forecast", "prediction", "actual_vs_predicted"])]
    if forecast_files:
        selected_path = st.selectbox("Forecast output", forecast_files, format_func=lambda p: str(Path(p).relative_to(ROOT)))
        forecast_df = read_csv_safely(selected_path)
        st.success(f"Loaded {len(forecast_df):,} forecast rows.")
        st.dataframe(forecast_df.head(1000), use_container_width=True)
        st.download_button("Download forecast output", forecast_df.to_csv(index=False).encode("utf-8"), "stocksense_forecast.csv", "text/csv")
    else:
        chart_message("No forecast output CSV detected yet. Train and save the demand forecasting model, then place its output under data/processed or reports. This page intentionally does not invent predictions.")

elif page == "Stockout Prediction":
    render_header("Stockout risk intelligence", "Review saved stockout predictions and model evaluation artifacts.")
    risk_files = [p for p in csv_paths if any(k in Path(p).name.lower() for k in ["stockout", "risk_prediction", "risk_score"])]
    if risk_files:
        selected_path = st.selectbox("Stockout output", risk_files, format_func=lambda p: str(Path(p).relative_to(ROOT)))
        risk_df = read_csv_safely(selected_path)
        st.success(f"Loaded {len(risk_df):,} stockout-related rows.")
        st.dataframe(risk_df.head(1000), use_container_width=True)
        st.download_button("Download stockout output", risk_df.to_csv(index=False).encode("utf-8"), "stocksense_stockout_predictions.csv", "text/csv")
    else:
        chart_message("No saved stockout prediction output was detected. This page will display real predictions once the trained model output is available.")

elif page == "Inventory Optimization":
    render_header("Inventory optimization", "Explore reorder recommendations and inventory policy outputs.")
    recommendation_files = [p for p in csv_paths if any(k in Path(p).name.lower() for k in ["reorder_recommendation", "reorder", "inventory_policy"])]
    if recommendation_files:
        selected_path = st.selectbox("Recommendation output", recommendation_files, format_func=lambda p: str(Path(p).relative_to(ROOT)))
        rec_df = read_csv_safely(selected_path)
        st.success(f"Loaded {len(rec_df):,} recommendation rows.")
        st.dataframe(rec_df.head(1500), use_container_width=True)
        st.download_button("Download reorder recommendations", rec_df.to_csv(index=False).encode("utf-8"), "stocksense_reorder_recommendations.csv", "text/csv")
    else:
        chart_message("No reorder recommendation CSV was detected. Check that the inventory optimization pipeline has generated its output.")

elif page == "Procurement Intelligence":
    render_header("Procurement intelligence", "Supplier, purchasing, receiving, and invoice analytics — separate from retail SKU analysis.")
    if procurement.empty:
        # Try showing all procurement-related CSVs as individual selectable sources.
        proc_paths = [p for p in csv_paths if any(k in Path(p).name.lower() for k in ["supplier", "purchase", "invoice", "receipt", "requisition", "buyer", "po_line", "items"])]
        if proc_paths:
            selected_path = st.selectbox("Procurement source file", proc_paths, format_func=lambda p: str(Path(p).relative_to(ROOT)))
            proc_df = read_csv_safely(selected_path)
            st.write(f"{len(proc_df):,} rows × {len(proc_df.columns)} columns")
            st.dataframe(proc_df.head(1000), use_container_width=True)
        else:
            chart_message("No procurement CSV files detected.")
    else:
        st.write(f"Loaded procurement dataset: {Path(procurement_path).name}")
        st.dataframe(procurement.head(1000), use_container_width=True)

elif page == "Model Performance":
    render_header("Model performance center", "Metrics and evaluation outputs generated by the actual training pipeline.")
    metric_files = [p for p in csv_paths if any(k in Path(p).name.lower() for k in ["metric", "evaluation", "backtest", "performance", "comparison"])]
    if metric_files:
        selected_path = st.selectbox("Evaluation file", metric_files, format_func=lambda p: str(Path(p).relative_to(ROOT)))
        metrics_df = read_csv_safely(selected_path)
        st.dataframe(metrics_df, use_container_width=True)
        st.download_button("Download evaluation CSV", metrics_df.to_csv(index=False).encode("utf-8"), "stocksense_model_metrics.csv", "text/csv")
    else:
        chart_message("No model evaluation CSV detected. Train and evaluate the forecasting and stockout models to populate this page.")

elif page == "About StockSense":
    render_header("About StockSense", "Supply Chain Intelligence and Inventory Optimization.")
    st.markdown("""
    <div class="glass">
      <h3 style="color:#eaf5ff;margin-top:0;">From operational data to better decisions.</h3>
      <p style="color:#a9bfd8;line-height:1.7;">
      StockSense is an end-to-end data science and machine learning project for retail
      sales analysis, inventory optimization, demand forecasting, stockout-risk analysis,
      and independent procurement analytics.
      </p>
      <p style="color:#8fa8c4;font-size:.85rem;">
      This dashboard starter reads local CSV artifacts from the existing project. Forecast,
      stockout, and model-performance pages will populate when the corresponding pipeline
      outputs are available.
      </p>
    </div>
    """, unsafe_allow_html=True)
    show_data_status()

st.markdown(
    f"""<div style="border-top:1px solid rgba(87,190,255,.15);margin-top:35px;padding-top:14px;
    color:#6f89a5;font-size:11px;display:flex;justify-content:space-between;">
    <span>STOCKSENSE · SUPPLY CHAIN INTELLIGENCE</span>
    <span>Local project workspace · Real data only</span></div>""",
    unsafe_allow_html=True,
)
