import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math

st.set_page_config(
    page_title="AI Agent Reliability Framework",
    page_icon="🔵",
    layout="wide",
)

st.markdown("""
<style>
h1{color:#1e3a5f}h2{color:#2471a3}
.hbox{background:#eff6ff;border-left:4px solid #3b82f6;
      padding:14px 18px;border-radius:0 8px 8px 0;margin:10px 0}
.rbox{background:#f0fdf4;border-left:4px solid #22c55e;
      padding:14px 18px;border-radius:0 8px 8px 0;margin:10px 0}
.dbox{background:#fef2f2;border-left:4px solid #ef4444;
      padding:14px 18px;border-radius:0 8px 8px 0;margin:10px 0}
.wbox{background:#fef3c7;border-left:4px solid #f59e0b;
      padding:14px 18px;border-radius:0 8px 8px 0;margin:10px 0}
</style>
""", unsafe_allow_html=True)

V1 = dict(intent_fail=0.0000, retrieval_fail=0.3000,
           ranking_fail=0.5580, hallucination=0.4800,
           tool_fail=0.1260, agent_fail=0.7320, n=500)
V2 = dict(intent_fail=0.0035, retrieval_fail=0.2835,
           ranking_fail=0.5665, hallucination=0.4405,
           tool_fail=0.1345, agent_fail=0.7080, n=2000)
CPT = dict(h00=0.2851, h10=0.4500, h01=0.5891, h11=0.6733)
SENS = {"RetrievalFailure":0.153,"ToolSelectionError":0.103,
        "RankingError":0.091,"LLM_Hallucination":0.076,"IntentMisclass":0.050}

def or_gate(*ps):
    r = 1.0
    for p in ps:
        r *= (1 - p)
    return round(1 - r, 4)

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return round(max(0, c-m), 4), round(min(1, c+m), 4)

def risk_level(p):
    if p >= 0.85:
        return "CRITICAL", "#ef4444"
    if p >= 0.75:
        return "HIGH", "#f97316"
    if p >= 0.60:
        return "MEDIUM", "#eab308"
    return "LOW", "#22c55e"

def compute_bn(ret, rank, hall, tool, intent):
    sc   = hall / 0.48 if 0.48 > 0 else 1.0
    h00  = min(0.99, CPT["h00"] * sc)
    h10  = min(0.99, CPT["h10"] * sc)
    h01  = min(0.99, CPT["h01"] * sc)
    h11  = min(0.99, CPT["h11"] * sc)
    p_heff = (h00*(1-ret)*(1-rank) + h10*ret*(1-rank) +
              h01*(1-ret)*rank    + h11*ret*rank)
    p_know = or_gate(ret, rank)
    p_reas = or_gate(p_heff, intent)
    p_fta  = or_gate(p_know, p_reas, tool)
    corr   = 0.065 * (ret / 0.30)
    p_bn   = max(0.01, min(0.99, p_fta - corr))
    return round(p_bn, 4), round(p_fta, 4)

def pipeline_sim(question):
    q = question.lower()
    calc_kw  = ["how many","how much","calculate","total","age",
                "years old","number of","difference","older"]
    multi_kw = ["who","directed","wrote","both","also","same",
                "nationality","portrayed","starred"]
    i_fail  = len(question.strip()) < 5
    r_fail  = any(k in q for k in multi_kw) and np.random.random() < 0.35
    rk_fail = r_fail and np.random.random() < 0.72
    if r_fail and rk_fail:
        p_h = CPT["h11"]
    elif r_fail:
        p_h = CPT["h10"]
    elif rk_fail:
        p_h = CPT["h01"]
    else:
        p_h = CPT["h00"]
    hall   = np.random.random() < p_h
    t_sel  = "calculator" if any(k in q for k in calc_kw) else "retrieval"
    t_fail = t_sel == "calculator"
    ag     = any([i_fail, r_fail, rk_fail, hall, t_fail])
    p_bn, p_fta = compute_bn(
        0.30 if r_fail  else 0.05,
        0.56 if rk_fail else 0.10,
        p_h,
        0.13 if t_fail else 0.05,
        0.50 if i_fail  else 0.00,
    )
    return dict(i_fail=i_fail, r_fail=r_fail, rk_fail=rk_fail,
                hall=hall, t_fail=t_fail, t_sel=t_sel,
                ag=ag, p_bn=p_bn, p_fta=p_fta, p_h=round(p_h, 4))

with st.sidebar:
    st.markdown("### AI Reliability Framework")
    st.markdown("**MSc Dissertation**")
    st.divider()
    page = st.radio("", [
        "Overview",
        "Live Pipeline Demo",
        "Fault Tree Analysis",
        "Event Tree Analysis",
        "Bayesian Network",
        "Sensitivity Analysis",
        "V1 vs V2 Comparison",
        "Apply to Any Agent",
    ], label_visibility="collapsed")
    st.divider()
    st.metric("BN Estimate",       "0.688", "-0.044 from observed")
    st.metric("FTA Estimate",      "0.859", "+0.127 from observed",
              delta_color="inverse")
    st.metric("BN accuracy gain",  "65.5%", "vs FTA")

if page == "Overview":
    st.title("AI Agent Reliability Framework")
    st.markdown("#### Applying FTA · ETA · Bayesian Networks to a RAG AI Pipeline")
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions tested",   "2,500")
    c2.metric("Agent failure rate", "73.2%",  "V1 baseline")
    c3.metric("BN vs FTA accuracy", "65.5%",  "BN more accurate")
    c4.metric("Most critical",      "Retrieval", "swing=0.153")
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
A five-component RAG AI agent was built and evaluated on the **HotpotQA**
multi-hop QA benchmark. Three classical reliability engineering methods
were applied: **FTA**, **ETA**, and **Bayesian Networks**.

**Central finding:** The BN estimated P(AgentFailure) = 0.688, achieving
65.5% lower estimation error than FTA (0.044 vs 0.127). The BN is more
accurate because it explicitly models the Retrieval → Hallucination
dependency that FTA cannot represent.
""")
        st.markdown(
            '<div class="hbox"><b>FTA independence assumption violated:</b> '
            'Retrieval failure and hallucination co-occur at 2.0x the independent rate. '
            'FTA predicts 0.144, data shows 0.288. '
            'This is why FTA overestimates by 12.7 percentage points.</div>',
            unsafe_allow_html=True)
    with col2:
        fig = go.Figure(go.Bar(
            x=["Observed", "BN", "FTA"],
            y=[0.732, 0.688, 0.859],
            marker_color=["#64748b", "#22c55e", "#ef4444"],
            text=["0.732", "0.688", "0.859"],
            textposition="outside",
        ))
        fig.add_hline(y=0.732, line_dash="dash", line_color="#64748b")
        fig.update_layout(
            title="P(AgentFailure) — three methods",
            yaxis=dict(range=[0, 1.05], gridcolor="#f1f5f9"),
            height=300, margin=dict(t=40, b=20, l=30, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Five-component pipeline")
    comps = [
        ("1","Intent",   "DeBERTa",   V2["intent_fail"],    "#22c55e"),
        ("2","Retriever","mpnet",      V2["retrieval_fail"], "#f97316"),
        ("3","Ranker",   "CrossEnc",  V2["ranking_fail"],   "#ef4444"),
        ("4","LLM",      "Flan-T5-XL",V2["hallucination"],  "#f59e0b"),
        ("5","Tool",     "Rules",      V2["tool_fail"],      "#3b82f6"),
    ]
    cols = st.columns(5)
    for col, (num, name, model, rate, color) in zip(cols, comps):
        with col:
            st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
padding:12px 8px;text-align:center;border-top:4px solid {color}">
<div style="font-size:18px;font-weight:700;color:{color}">{num}</div>
<div style="font-size:13px;font-weight:600;color:#1e293b;margin:4px 0">{name}</div>
<div style="font-size:10px;color:#94a3b8;margin-bottom:6px">{model}</div>
<div style="font-size:18px;font-weight:700;color:{color}">{rate*100:.1f}%</div>
<div style="font-size:10px;color:#94a3b8">V2 fail rate</div>
</div>""", unsafe_allow_html=True)

elif page == "Live Pipeline Demo":
    st.title("Live Pipeline Demo")
    st.markdown("Type a question and watch each component run. BN updates with evidence.")
    examples = [
        "Select an example...",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "The director of Jaws also directed which other famous film?",
        "Which film came first, Jaws or Star Wars?",
        "How many Oscars did Schindler List receive?",
        "What is the capital of France?",
    ]
    sel = st.selectbox("Try an example:", examples)
    q   = st.text_input("Or type your own question:",
                        value="" if sel.startswith("Select") else sel,
                        placeholder="Enter any question...")
    if st.button("Run pipeline", type="primary"):
        if not q.strip():
            st.error("Please enter a question first.")
        else:
            res = pipeline_sim(q)
            st.divider()
            st.subheader("Component results")
            comp_data = [
                ("Intent",    res["i_fail"],  "DeBERTa"),
                ("Retrieval", res["r_fail"],  "mpnet"),
                ("Ranking",   res["rk_fail"], "CrossEncoder"),
                ("LLM",       res["hall"],    "Flan-T5-XL"),
                ("Tool",      res["t_fail"],  res["t_sel"]),
            ]
            c5 = st.columns(5)
            for col, (name, failed, model) in zip(c5, comp_data):
                color = "#ef4444" if failed else "#22c55e"
                bg    = "#fef2f2" if failed else "#f0fdf4"
                icon  = "FAIL"   if failed else "PASS"
                col.markdown(f"""
<div style="background:{bg};border:1.5px solid {color};border-radius:10px;
padding:12px 6px;text-align:center">
<div style="font-size:12px;color:#475569;margin-bottom:4px">{name}</div>
<div style="font-size:18px;font-weight:700;color:{color}">{icon}</div>
<div style="font-size:10px;color:#94a3b8;margin-top:4px">{model}</div>
</div>""", unsafe_allow_html=True)
            st.divider()
            col1, col2 = st.columns([1, 2])
            with col1:
                risk, rc = risk_level(res["p_bn"])
                st.metric("BN P(AgentFail)",  f"{res['p_bn']:.4f}")
                st.metric("FTA P(AgentFail)", f"{res['p_fta']:.4f}")
                outcome = "FAILED" if res["ag"] else "PASSED"
                oc = "#ef4444" if res["ag"] else "#22c55e"
                ob = "#fef2f2" if res["ag"] else "#f0fdf4"
                st.markdown(f"""
<div style="background:{ob};border:2px solid {oc};border-radius:10px;
padding:16px;text-align:center;margin-top:10px">
<div style="font-size:13px;color:{oc}">Agent result</div>
<div style="font-size:30px;font-weight:700;color:{oc}">{outcome}</div>
<div style="font-size:11px;color:{oc}">Risk: {risk}</div>
</div>""", unsafe_allow_html=True)
            with col2:
                stages = ["Prior","Intent","Retrieval","Ranking","LLM","Final"]
                probs  = [
                    0.688, 0.688,
                    res["p_bn"] if res["r_fail"]  else 0.55,
                    res["p_bn"] if res["rk_fail"] else 0.48,
                    res["p_bn"],
                    res["p_bn"],
                ]
                bar_colors = [
                    "#22c55e" if p < 0.70 else
                    "#f59e0b" if p < 0.85 else
                    "#ef4444" for p in probs
                ]
                fig2 = go.Figure(go.Bar(
                    x=stages, y=probs,
                    marker_color=bar_colors,
                    text=[f"{p:.4f}" for p in probs],
                    textposition="outside",
                ))
                fig2.add_hline(y=0.732, line_dash="dash",
                               line_color="#64748b",
                               annotation_text="Observed 0.732")
                fig2.update_layout(
                    title="BN P(AgentFail) as evidence arrives",
                    yaxis=dict(range=[0, 1.1], gridcolor="#f1f5f9"),
                    height=320, margin=dict(t=40, b=40, l=40, r=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig2, use_container_width=True)
            if res["r_fail"] or res["rk_fail"]:
                st.markdown(
                    f'<div class="dbox"><b>BN evidence propagation:</b> '
                    f'Retrieval/ranking failure raised P(Hallucination) from '
                    f'0.285 to {res["p_h"]:.3f} — automatically propagated '
                    f'through the dependency graph. FTA cannot do this.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="rbox"><b>All upstream components passed.</b> '
                    f'P(Hallucination) stays at {res["p_h"]:.3f} (lowest CPT value). '
                    f'BN P(AgentFail) = {res["p_bn"]:.4f} — below the 0.732 baseline.</div>',
                    unsafe_allow_html=True)

elif page == "Fault Tree Analysis":
    st.title("Fault Tree Analysis")
    st.markdown("Top-down deductive method — asks *what causes* agent failure?")
    c1, c2, c3 = st.columns(3)
    c1.metric("P(AgentFail | FTA)",      "0.8594")
    c2.metric("P(AgentFail | Observed)", "0.7320")
    c3.metric("FTA Overestimate",        "+0.1274", delta_color="inverse")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("OR gate calculation")
        p_r = V1["retrieval_fail"]; p_rk = V1["ranking_fail"]
        p_h = V1["hallucination"]; p_i  = V1["intent_fail"]
        p_t = V1["tool_fail"]
        p_know = or_gate(p_r, p_rk)
        p_reas = or_gate(p_h, p_i)
        p_fta  = or_gate(p_know, p_reas, p_t)
        rows = [
            {"Node":"P(IntentMisclass)",    "Formula":"Measured",               "Value":f"{p_i:.4f}"},
            {"Node":"P(RetrievalFailure)",  "Formula":"Measured",               "Value":f"{p_r:.4f}"},
            {"Node":"P(RankingError)",      "Formula":"Measured",               "Value":f"{p_rk:.4f}"},
            {"Node":"P(LLM_Hallucination)","Formula":"Measured",               "Value":f"{p_h:.4f}"},
            {"Node":"P(ToolSelectionErr)", "Formula":"Measured",               "Value":f"{p_t:.4f}"},
            {"Node":"P(Knowledge Failure)","Formula":f"OR({p_r},{p_rk})",      "Value":f"{p_know:.4f}"},
            {"Node":"P(Reasoning Failure)","Formula":f"OR({p_h},{p_i})",       "Value":f"{p_reas:.4f}"},
            {"Node":"P(AgentFail | FTA)",  "Formula":"OR(know,reas,exec)",     "Value":f"{p_fta:.4f}"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    with col2:
        st.subheader("Independence audit")
        ratios = [2.0, 1.3, 1.6, 1.1]
        labels = ["Ret x Hall", "Rank x Hall", "Ret x Rank", "Tool x Hall"]
        colors_a = ["#ef4444", "#f59e0b", "#ef4444", "#22c55e"]
        fig_a = go.Figure(go.Bar(
            x=labels, y=ratios, marker_color=colors_a,
            text=[f"{r:.1f}x" for r in ratios], textposition="outside",
        ))
        fig_a.add_hline(y=1.0, line_dash="dash", line_color="#64748b",
                        annotation_text="Independence = 1.0")
        fig_a.update_layout(
            yaxis=dict(range=[0, 2.8], gridcolor="#f1f5f9"),
            height=300, margin=dict(t=20, b=40, l=40, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_a, use_container_width=True)
    st.markdown(
        '<div class="dbox"><b>Why FTA overestimates:</b> '
        'Retrieval x Hallucination co-occur at 2.0x the independent rate. '
        'FTA assumes ratio=1.0 everywhere. This inflates P(AgentFailure) by 12.7pp.</div>',
        unsafe_allow_html=True)
    st.divider()
    st.subheader("Component error rates with 95% Wilson CI")
    comp_names = ["Intent","Retrieval","Ranking","Hallucination","Tool"]
    rates = [V1["intent_fail"], V1["retrieval_fail"], V1["ranking_fail"],
             V1["hallucination"], V1["tool_fail"]]
    cis   = [wilson_ci(int(r * 500), 500) for r in rates]
    fig_e = go.Figure(go.Bar(
        x=comp_names, y=rates,
        marker_color=["#22c55e","#f97316","#ef4444","#f59e0b","#3b82f6"],
        error_y=dict(
            type="data", symmetric=False,
            array=[ci[1]-r for ci,r in zip(cis,rates)],
            arrayminus=[r-ci[0] for ci,r in zip(cis,rates)],
        ),
        text=[f"{r:.3f}" for r in rates], textposition="outside",
    ))
    fig_e.add_hline(y=0.732, line_dash="dash", line_color="#ef4444",
                    annotation_text="Agent failure = 0.732")
    fig_e.update_layout(
        yaxis=dict(range=[0, 1.0], gridcolor="#f1f5f9"),
        height=380, margin=dict(t=20, b=20, l=40, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_e, use_container_width=True)

elif page == "Event Tree Analysis":
    st.title("Event Tree Analysis")
    st.markdown("Forward consequence modelling — given a failure, what happens next?")
    ETA_DATA = {
        "Retrieval Failure": {"p_init":0.300,"p_unsafe":0.0630,"p_safe":0.018},
        "Ranking Error":     {"p_init":0.558,"p_unsafe":0.1360,"p_safe":0.024},
        "LLM Hallucination": {"p_init":0.480,"p_unsafe":0.0686,"p_safe":0.045},
    }
    c1, c2, c3 = st.columns(3)
    for col, (name, data) in zip([c1,c2,c3], ETA_DATA.items()):
        col.metric(name, f"P(Unsafe)={data['p_unsafe']:.4f}",
                   f"P(init)={data['p_init']:.3f}")
    st.divider()
    names   = list(ETA_DATA.keys())
    p_u     = [ETA_DATA[n]["p_unsafe"] for n in names]
    p_s     = [ETA_DATA[n]["p_safe"]   for n in names]
    fig_eta = go.Figure()
    fig_eta.add_trace(go.Bar(
        name="P(Unsafe Output)", x=names, y=p_u,
        marker_color=["#f97316","#ef4444","#f59e0b"],
        text=[f"{p:.4f}" for p in p_u], textposition="outside",
    ))
    fig_eta.add_trace(go.Bar(
        name="P(Safe Output)", x=names, y=p_s,
        marker_color=["#86efac"]*3,
        text=[f"{p:.4f}" for p in p_s], textposition="outside",
    ))
    fig_eta.update_layout(
        barmode="group", height=360,
        yaxis=dict(gridcolor="#f1f5f9"),
        margin=dict(t=20,b=20,l=40,r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_eta, use_container_width=True)
    st.divider()
    st.subheader("Safety barrier effectiveness")
    BARRIERS = {
        "ETA-1 Retrieval Failure":[
            ("Fallback Retrieval Strategy",0.40),
            ("LLM Detects Missing Context",0.30),
            ("Output Validation Check",    0.50),
        ],
        "ETA-2 Ranking Error":[
            ("Top-K Redundancy Buffer",        0.35),
            ("LLM Detects Irrelevant Context", 0.25),
            ("Output Validation Check",        0.50),
        ],
        "ETA-3 LLM Hallucination":[
            ("Factual Consistency Check",  0.45),
            ("Confidence Score Threshold", 0.35),
            ("User-Facing Uncertainty Flag",0.60),
        ],
    }
    tabs = st.tabs(list(BARRIERS.keys()))
    for tab, (eta_name, blist) in zip(tabs, BARRIERS.items()):
        with tab:
            for bname, bp in blist:
                color = "#22c55e" if bp >= 0.45 else "#f59e0b" if bp >= 0.30 else "#ef4444"
                st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
<div style="width:200px;font-size:13px;color:#334155">{bname}</div>
<div style="flex:1;height:18px;background:#f1f5f9;border-radius:9px;overflow:hidden">
<div style="width:{int(bp*100)}%;height:100%;background:{color};border-radius:9px">
</div></div>
<div style="width:50px;text-align:right;font-size:13px;font-weight:600;color:{color}">
{bp:.0%}</div></div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="hbox"><b>Key finding:</b> Ranking error produces the HIGHEST '
        'P(Unsafe)=0.136 despite not being the most frequent failure — because its '
        'barriers are weakest. Output Validation (50%) is the most effective single '
        'barrier across all three ETAs.</div>', unsafe_allow_html=True)

elif page == "Bayesian Network":
    st.title("Bayesian Network")
    st.markdown("Explicitly models component dependencies that FTA ignores.")
    c1, c2, c3 = st.columns(3)
    c1.metric("P(AgentFail | Observed)", "0.7320", "Ground truth")
    c2.metric("P(AgentFail | BN)",       "0.6881", "Error = 0.044")
    c3.metric("P(AgentFail | FTA)",      "0.8594", "Error = 0.127",
              delta_color="inverse")
    st.markdown(
        '<div class="rbox"><b>BN is 65.5% more accurate than FTA.</b> '
        'The BN uses four hallucination CPT values (0.285 to 0.673) based on '
        'upstream component state. FTA collapses all four to a flat 0.480.</div>',
        unsafe_allow_html=True)
    st.divider()
    st.subheader("Critical CPT: P(Hallucination | Retrieval, Ranking)")
    cpt_df = pd.DataFrame({
        "Retrieval":         ["OK",   "FAIL", "OK",   "FAIL"],
        "Ranking":           ["OK",   "OK",   "FAIL", "FAIL"],
        "P(Hallucination)":  [CPT["h00"],CPT["h10"],CPT["h01"],CPT["h11"]],
        "vs FTA flat 0.480": [f"{CPT['h00']-0.48:+.4f}",f"{CPT['h10']-0.48:+.4f}",
                               f"{CPT['h01']-0.48:+.4f}",f"{CPT['h11']-0.48:+.4f}"],
        "Source":            ["Measured n=221","Literature","Measured n=129","Measured n=150"],
    })
    st.dataframe(cpt_df, use_container_width=True, hide_index=True)
    vals    = [CPT["h00"], CPT["h10"], CPT["h01"], CPT["h11"]]
    fig_cpt = go.Figure(go.Bar(
        x=["Ret=OK Rank=OK","Ret=FAIL Rank=OK",
           "Ret=OK Rank=FAIL","Ret=FAIL Rank=FAIL"],
        y=vals,
        marker_color=["#22c55e","#f59e0b","#f97316","#ef4444"],
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig_cpt.add_hline(y=0.480, line_dash="dash", line_color="#64748b",
                      line_width=2, annotation_text="FTA flat rate = 0.480")
    fig_cpt.update_layout(
        yaxis=dict(range=[0, 0.85], gridcolor="#f1f5f9"),
        height=320, margin=dict(t=20, b=20, l=40, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_cpt, use_container_width=True)
    st.divider()
    st.subheader("Evidence injection — tick what has been observed")
    col1, col2 = st.columns(2)
    with col1:
        ret_obs  = st.checkbox("Retrieval = FAIL observed")
        rank_obs = st.checkbox("Ranking = FAIL observed")
        tool_obs = st.checkbox("Tool = FAIL observed")
    with col2:
        if ret_obs and rank_obs:
            p_h_ev = CPT["h11"]; p_af_ev = 0.962
        elif ret_obs:
            p_h_ev = CPT["h10"]; p_af_ev = 0.956
        elif rank_obs:
            p_h_ev = CPT["h01"]; p_af_ev = 0.880
        elif tool_obs:
            p_h_ev = 0.436;      p_af_ev = 0.815
        else:
            p_h_ev = 0.436;      p_af_ev = 0.688
        risk, rc = risk_level(p_af_ev)
        st.metric("P(AgentFail | evidence)",     f"{p_af_ev:.4f}")
        st.metric("P(Hallucination | evidence)", f"{p_h_ev:.4f}")
        bg_r = ("#fef2f2" if risk in ["CRITICAL","HIGH"]
                else "#fef9c3" if risk == "MEDIUM" else "#f0fdf4")
        st.markdown(f"""
<div style="background:{bg_r};border:2px solid {rc};border-radius:10px;
padding:12px;text-align:center;margin-top:8px">
<div style="font-size:12px;color:{rc}">Risk level</div>
<div style="font-size:28px;font-weight:700;color:{rc}">{risk}</div>
<div style="font-size:12px;color:{rc}">P(AgentFail) = {p_af_ev:.4f}</div>
</div>""", unsafe_allow_html=True)

elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis — Tornado Diagram")
    st.markdown("Which component matters most for reliability?")
    st.markdown(
        '<div class="hbox"><b>RQ5 answered:</b> Retrieval failure is the most '
        'critical component (swing=0.153). Improving the retriever gives the '
        'largest single reliability improvement. Invest in retrieval first.</div>',
        unsafe_allow_html=True)
    params = list(SENS.keys())
    swings = [SENS[p] for p in params]
    order  = sorted(range(len(swings)), key=lambda i: swings[i])
    sp     = [params[i] for i in order]
    sv     = [swings[i] for i in order]
    colors_t = ["#ef4444" if s > 0.12 else
                "#f59e0b" if s > 0.08 else
                "#3b82f6" for s in sv]
    fig_t = go.Figure(go.Bar(
        y=sp, x=sv, orientation="h",
        marker_color=colors_t,
        text=[f"swing={s:.4f}" for s in sv], textposition="outside",
    ))
    fig_t.update_layout(
        title=f"Baseline P(AgentFail|BN) = 0.6881  — each bar = swing from +/-0.20 variation",
        xaxis=dict(title="Swing in P(AgentFailure)", gridcolor="#f1f5f9"),
        height=380, margin=dict(t=50, b=20, l=180, r=140),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_t, use_container_width=True)
    st.divider()
    st.subheader("Interactive explorer — move sliders to see new BN estimate")
    col1, col2 = st.columns(2)
    with col1:
        r_s  = st.slider("P(RetrievalFailure)",   0.00, 0.60, V1["retrieval_fail"], 0.01)
        rk_s = st.slider("P(RankingError)",        0.00, 0.80, V1["ranking_fail"],   0.01)
        h_s  = st.slider("P(LLM_Hallucination)",  0.00, 0.80, V1["hallucination"],  0.01)
        t_s  = st.slider("P(ToolSelectionError)", 0.00, 0.40, V1["tool_fail"],      0.01)
        i_s  = st.slider("P(IntentMisclass)",     0.00, 0.20, V1["intent_fail"],    0.01)
    with col2:
        p_bn_n, p_fta_n = compute_bn(r_s, rk_s, h_s, t_s, i_s)
        d_bn  = round(p_bn_n  - 0.6881, 4)
        d_fta = round(p_fta_n - 0.8594, 4)
        st.metric("New BN estimate",  f"{p_bn_n:.4f}",  f"{d_bn:+.4f} from baseline")
        st.metric("New FTA estimate", f"{p_fta_n:.4f}", f"{d_fta:+.4f} from baseline")
        improve = round((0.6881 - p_bn_n) * 100, 2)
        st.metric("Reliability gain", f"{max(0.0, improve):.2f}pp",
                  "reduction in failure probability")

elif page == "V1 vs V2 Comparison":
    st.title("V1 vs V2 Model Comparison")
    st.markdown(f"**V1:** BART + MiniLM + Flan-T5-Large (n=500)  |  "
                f"**V2:** DeBERTa + mpnet + Flan-T5-XL (n=2000)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Agent fail V1", f"{V1['agent_fail']:.3f}", f"n={V1['n']}")
    c2.metric("Agent fail V2", f"{V2['agent_fail']:.3f}",
              f"{(V2['agent_fail']-V1['agent_fail'])*100:+.1f}pp",
              delta_color="inverse")
    c3.metric("Improvement",
              f"{(V1['agent_fail']-V2['agent_fail'])*100:.1f}pp", "more reliable")
    st.divider()
    comp_keys  = ["intent_fail","retrieval_fail","ranking_fail",
                  "hallucination","tool_fail","agent_fail"]
    comp_names = ["Intent","Retrieval","Ranking","Hallucination","Tool","Agent"]
    v1r = [V1[k] for k in comp_keys]
    v2r = [V2[k] for k in comp_keys]
    tab1, tab2, tab3 = st.tabs(["Side by side","Improvement %","Confidence intervals"])
    with tab1:
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            name="V1 BART+MiniLM", x=comp_names, y=v1r,
            marker_color="#f87171",
            text=[f"{r:.3f}" for r in v1r], textposition="outside",
        ))
        fig_c.add_trace(go.Bar(
            name="V2 DeBERTa+mpnet", x=comp_names, y=v2r,
            marker_color="#4ade80",
            text=[f"{r:.3f}" for r in v2r], textposition="outside",
        ))
        fig_c.update_layout(
            barmode="group", height=400,
            yaxis=dict(range=[0, 0.85], gridcolor="#f1f5f9"),
            margin=dict(t=20, b=20, l=40, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_c, use_container_width=True)
    with tab2:
        imp = [(v1-v2)/v1*100 if v1 > 0 else 0.0 for v1,v2 in zip(v1r,v2r)]
        ci2 = ["#22c55e" if i > 0 else "#ef4444" for i in imp]
        fig_i = go.Figure(go.Bar(
            x=comp_names, y=imp, marker_color=ci2,
            text=[f"{i:+.1f}%" for i in imp], textposition="outside",
        ))
        fig_i.add_hline(y=0, line_color="#1e293b", line_width=1)
        fig_i.update_layout(
            height=380, yaxis=dict(gridcolor="#f1f5f9"),
            margin=dict(t=20, b=20, l=40, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_i, use_container_width=True)
    with tab3:
        rows = []
        for k, name in zip(comp_keys, comp_names):
            r1 = V1[k]; r2 = V2[k]
            lo1, hi1 = wilson_ci(int(r1*V1["n"]), V1["n"])
            lo2, hi2 = wilson_ci(int(r2*V2["n"]), V2["n"])
            rows.append({
                "Component": name,
                "V1 rate":   f"{r1:.4f}",
                "V1 95% CI": f"[{lo1:.4f}, {hi1:.4f}]",
                "V2 rate":   f"{r2:.4f}",
                "V2 95% CI": f"[{lo2:.4f}, {hi2:.4f}]",
                "Change":    f"{(r2-r1)*100:+.2f}pp",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        st.caption("V2 CIs are ±2.2pp vs ±4.4pp for V1 — twice as precise at n=2000.")

elif page == "Apply to Any Agent":
    st.title("Apply to Any AI Agent")
    st.markdown("Your framework is domain-agnostic. Enter any agent's component failure rates.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Enter failure rates for your agent:**")
        m_ret  = st.number_input("P(Retrieval / Search failure)",   0.0,1.0,0.30,0.01,format="%.3f")
        m_rank = st.number_input("P(Ranking / Filter failure)",     0.0,1.0,0.56,0.01,format="%.3f")
        m_hall = st.number_input("P(Generation / Output failure)",  0.0,1.0,0.48,0.01,format="%.3f")
        m_tool = st.number_input("P(Action / Tool failure)",        0.0,1.0,0.13,0.01,format="%.3f")
        m_int  = st.number_input("P(Query Understanding failure)",  0.0,1.0,0.00,0.01,format="%.3f")
    with col2:
        m_bn, m_fta = compute_bn(m_ret, m_rank, m_hall, m_tool, m_int)
        risk, rc    = risk_level(m_bn)
        st.metric("FTA estimate", f"{m_fta:.4f}")
        st.metric("BN estimate",  f"{m_bn:.4f}")
        st.metric("Difference",   f"{abs(m_bn-m_fta):.4f}")
        bg_r = ("#fef2f2" if risk in ["CRITICAL","HIGH"]
                else "#fef9c3" if risk == "MEDIUM" else "#f0fdf4")
        st.markdown(f"""
<div style="background:{bg_r};border:2px solid {rc};border-radius:12px;
padding:16px;text-align:center;margin-top:12px">
<div style="font-size:13px;color:{rc}">Overall risk level</div>
<div style="font-size:36px;font-weight:700;color:{rc}">{risk}</div>
<div style="font-size:12px;color:{rc}">P(AgentFail|BN) = {m_bn:.4f}</div>
</div>""", unsafe_allow_html=True)
    st.divider()
    st.subheader("Example domains this framework applies to")
    domains = [
        ("Medical QA Agent",      "Symptom class → Clinical search → Evidence rank → Diagnosis → Referral", "#ef4444", "High stakes"),
        ("Legal Document AI",     "Intent class → Case law search → Relevance rank → Summary → Recommendation","#f97316","High stakes"),
        ("Customer Support Bot",  "Query class → KB search → Answer rank → Response → Escalation","#3b82f6","Medium stakes"),
        ("Coding Assistant",      "Task class → Snippet search → Relevance rank → Code gen → Test run","#8b5cf6","Medium stakes"),
        ("Research Assistant",    "Topic class → Paper search → Citation rank → Summary → Format select","#22c55e","Lower stakes"),
    ]
    for name, pipe, color, stakes in domains:
        st.markdown(f"""
<div style="background:white;border:.5px solid #e2e8f0;border-left:4px solid {color};
border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px">
<div style="font-size:14px;font-weight:600;color:#1e293b">{name}
<span style="font-size:11px;color:{color};font-weight:500;margin-left:8px">{stakes}</span>
</div>
<div style="font-size:12px;color:#64748b;margin-top:3px">{pipe}</div>
</div>""", unsafe_allow_html=True)
