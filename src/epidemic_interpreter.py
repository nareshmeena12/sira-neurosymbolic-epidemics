"""
epidemic_interpreter.py

LLM-powered agent that reads evaluation results and audit metrics,
then answers questions about the discovered equations in plain language.

Uses the OpenAI API (gpt-4o-mini by default).
Maintains conversation history so follow-up questions work naturally.

Usage:
    from src.epidemic_interpreter import EpidemicInterpreter

    agent = EpidemicInterpreter("results/depth_study/depth_study_results.csv")
    agent.generate_report()
    agent.ask("Why did experiment 3 struggle with gamma recovery?")
    agent.policy_brief("COVID-19")
"""

import os
import warnings

import numpy as np
import pandas as pd
from openai import OpenAI

warnings.filterwarnings("ignore")

SYSTEM_PROMPT = """You are an expert epidemiologist analysing SIR equation discovery results.

Metric definitions — use these exact names:
- SPI: Structural Purity Index — correct equation terms, no spurious ones
- PSS: Parameter Stability Score — stability under noise
- CLD: Conservation Law Deviation — S+I+R=1 conservation
- FH: Forecasting Horizon — future prediction coverage
- OOD: Out-of-Distribution MAE — generalisation to unseen regimes
- CNT: Critical Noise Threshold — max absorbable sensor noise
- DST: Data Sparsity Tolerance — min data fraction needed
- ZER: Zero-Shot Efficiency Ratio — accuracy per simulation used
- TTD: Time To Discovery — seconds from data to discovered ODE

Rules:
- Maximum 200 words per response
- Use the exact metric names above — never rename them
- Lead with the single most important finding
- Be honest about weaknesses
"""


class EpidemicInterpreter:
    """
    Reads evaluation results and answers questions about them using GPT.
    Maintains full conversation history across turns.
    """

    def __init__(self, csv_path: str, api_key: str = None,
                 model: str = "gpt-4o-mini", verbose: bool = True):
        self.verbose  = verbose
        self.csv_path = csv_path
        self.model    = model
        self.history  = []

        api_key      = api_key or os.environ.get("OPENAI_API_KEY")
        self.client  = OpenAI(api_key=api_key)

        self.df      = pd.read_csv(csv_path)
        self.context = self._build_context()

        if verbose:
            df_ok = self.df[self.df["status"] == "ok"] if "status" in self.df.columns else self.df
            print(f"EpidemicInterpreter ready — {len(self.df)} experiments loaded ({len(df_ok)} valid)")

    def _build_context(self) -> str:
        df = self.df.copy()

        if "status" in df.columns:
            df = df[df["status"] == "ok"]

        def stat(col):
            if col not in df.columns:
                return "N/A"
            vals = df[col].dropna()
            if vals.empty:
                return "N/A"
            return (
                f"median={vals.median():.4f}  mean={vals.mean():.4f}  "
                f"std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}"
            )

        rows = []
        for _, r in df.head(50).iterrows():
            b_true = r.get("true_beta",     0)
            b_est  = r.get("est_beta",      0)
            b_err  = r.get("beta_err_pct",  0)
            g_true = r.get("true_gamma",    0)
            g_est  = r.get("est_gamma",     0)
            g_err  = r.get("gamma_err_pct", 0)

            try:
                row_str = (
                    f"  Exp{int(r.get('experiment_id', 0))+1:3d}: "
                    f"b={float(b_true):.3f}->{float(b_est):.3f} "
                    f"({float(b_err):.1f}%)  "
                    f"g={float(g_true):.3f}->{float(g_est):.3f} "
                    f"({float(g_err):.1f}%)"
                )
            except (ValueError, TypeError):
                continue

            if "spi_overall" in r and pd.notna(r["spi_overall"]):
                row_str += f"  SPI={r['spi_overall']:.3f}"
            if "cld_max" in r and pd.notna(r["cld_max"]):
                row_str += f"  CLD={r['cld_max']:.8f}"
            if "fh_pct" in r and pd.notna(r["fh_pct"]):
                row_str += f"  FH={r['fh_pct']:.1f}%"
            if "zer" in r and pd.notna(r["zer"]):
                row_str += f"  ZER={r['zer']:.3f}"
            rows.append(row_str)

        context = f"""
EVALUATION RESULTS
==================
Total experiments  : {len(self.df)}
Valid runs         : {len(df)}

PARAMETER RECOVERY
------------------
Beta  error (%)    : {stat('beta_err_pct')}
Gamma error (%)    : {stat('gamma_err_pct')}

STRUCTURAL PURITY
-----------------
SPI overall        : {stat('spi_overall')}
SPI precision      : {stat('spi_precision')}
SPI recall         : {stat('spi_recall')}

PHYSICS PRESERVATION
--------------------
CLD max deviation  : {stat('cld_max')}

FORECASTING
-----------
FH coverage (%)    : {stat('fh_pct')}
FH days            : {stat('fh_days')}

DATA EFFICIENCY
---------------
ZER                : {stat('zer')}

OOD GENERALISATION
------------------
OOD overall MAE    : {stat('ood_overall')}
OOD explosive      : {stat('ood_explosive')}
OOD fast dieout    : {stat('ood_fast_dieout')}
OOD high turnover  : {stat('ood_high_turn')}
OOD slow burn      : {stat('ood_slow_burn')}

PER-EXPERIMENT DETAIL (first 50)
---------------------------------
{chr(10).join(rows)}
"""
        return context

    def _call(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + self.context},
                *self.history,
            ],
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def ask(self, question: str) -> str:
        """Ask any question about the results. Conversation history is maintained."""
        if self.verbose:
            print(f"\nQ: {question}")
            print("-" * 50)
        reply = self._call(question)
        if self.verbose:
            print(reply)
        return reply

    def generate_report(self, save_path: str = "results/llm_report.md") -> str:
        """Generates a structured report and saves it to markdown."""
        prompt = """Write a structured technical report on these SIR equation
discovery results. Use these exact section headers:

## Executive Summary
## Parameter Recovery
## Equation Structural Quality
## Physics Preservation
## Forecasting and Generalisation
## Data Efficiency
## Limitations and Next Steps

Be direct. One paragraph per section max. No bullet points."""

        report = self._call(prompt)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write("# SIRA — Automated Analysis Report\n\n")
            f.write(report)

        if self.verbose:
            print("\n" + "=" * 55)
            print("AUTOMATED REPORT")
            print("=" * 55)
            print(report)
            print(f"\nSaved to {save_path}")

        return report

    def explain_experiment(self, experiment_idx: int) -> str:
     df_ok = self.df[self.df["status"] == "ok"].reset_index(drop=True) \
        if "status" in self.df.columns else self.df.reset_index(drop=True)

     row = df_ok.iloc[experiment_idx]
     return self.ask(
        f"In under 150 words, what are the two most important findings "
        f"from this experiment and what do they mean for epidemic surveillance?\n"
        f"{row.to_string()}"
    )

    def compare_experiments(self, metric: str = "beta_err_pct") -> str:
        """Compares best and worst experiments by the given metric."""
        df = self.df.copy()
        if "status" in df.columns:
            df = df[df["status"] == "ok"]

        if metric not in df.columns:
            return f"Column '{metric}' not found in CSV."

        cols = ["experiment_id", "true_beta", "true_gamma", metric]
        cols = [c for c in cols if c in df.columns]

        best  = df.nsmallest(5, metric)[cols].to_string(index=False)
        worst = df.nlargest(5, metric)[cols].to_string(index=False)

        return self.ask(
            f"Compare the 5 best and 5 worst experiments by {metric}.\n"
            f"Best:\n{best}\nWorst:\n{worst}\n"
            f"What epidemic characteristics separate them and what does "
            f"this tell us about where the pipeline is most and least reliable?"
        )

    def policy_brief(self, disease_name: str = "a novel respiratory virus") -> str:
        """Generates a 3-paragraph non-technical brief for public health officials."""
        return self.ask(
            f"Write a 3-paragraph non-technical policy brief as if these "
            f"results came from tracking {disease_name}. "
            f"Audience: public health ministers, not data scientists. "
            f"Cover what the model found, how reliable it is, "
            f"and what decisions it enables."
        )

    def reset(self):
        """Clears conversation history for a fresh session."""
        self.history = []
        if self.verbose:
            print("Conversation history cleared.")