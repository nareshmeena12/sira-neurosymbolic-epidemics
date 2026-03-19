"""
epidemic_interpreter.py

LLM-powered agent that reads evaluation results and audit metrics,
then answers questions about the discovered equations in plain language.

Uses the OpenAI API (gpt-4o-mini by default).
Maintains conversation history so follow-up questions work naturally.

Designed to demonstrate interpretation on 2-3 representative experiments,
not to summarise mass evaluation runs.

Usage:
    from src.epidemic_interpreter import EpidemicInterpreter

    agent = EpidemicInterpreter("results/evaluation_metrics.csv")
    agent.generate_report()
    agent.ask("Why did experiment 3 struggle with gamma recovery?")
    agent.ask("Is this pipeline ready for real outbreak surveillance?")
    agent.policy_brief("COVID-19")
"""

import os
import warnings

import numpy as np
import pandas as pd
from openai import OpenAI

warnings.filterwarnings("ignore")


SYSTEM_PROMPT = """You are an expert epidemiologist and machine learning scientist
specialising in equation discovery from stochastic epidemic data.

You are analysing results from a pipeline called SIRA that uses:
- Gillespie stochastic simulations to generate realistic epidemic data
- A neural network trained to learn a smooth continuous manifold over time
- PySINDy sparse regression to recover the governing SIR differential equations
- Nine evaluation metrics: SPI, PSS, CLD, FH, OOD MAE, CNT, DST, ZER, TTD

When interpreting results:
1. Lead with the most important finding
2. Explain metrics in plain language that a public health official could understand
3. Be honest about weaknesses, do not oversell
4. Focus on what the numbers mean for real epidemic surveillance
"""


class EpidemicInterpreter:
    """
    Reads evaluation results and answers questions about them using GPT.
    Maintains full conversation history across turns.

    Recommended usage: pick 2-3 representative experiments from your results
    (one good, one average, one hard case) and run the agent on those.
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
            n_valid = (self.df["Status"] == "ok").sum() if "Status" in self.df.columns else len(self.df)
            print(f"EpidemicInterpreter ready — {len(self.df)} experiments loaded ({n_valid} valid)")

    def _build_context(self) -> str:
        df = self.df.copy()

        if "Status" in df.columns:
            df = df[df["Status"] == "ok"]

        def stat(col):
            if col not in df.columns:
                return "N/A"
            vals = df[col].dropna()
            if vals.empty:
                return "N/A"
            return (f"median={vals.median():.4f}  mean={vals.mean():.4f}  "
                    f"std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

        rows = []
        for _, r in df.head(50).iterrows():
            row_str = (
                f"  Exp{int(r.get('Experiment_ID', 0))+1:3d}: "
                f"b={r.get('True_Beta','?'):.3f}->{r.get('Est_Beta','?'):.3f} "
                f"({r.get('Beta_Error_Pct','?'):.1f}%)  "
                f"g={r.get('True_Gamma','?'):.3f}->{r.get('Est_Gamma','?'):.3f} "
                f"({r.get('Gamma_Error_Pct','?'):.1f}%)"
            )
            if "spi_overall" in r:
                row_str += f"  SPI={r['spi_overall']:.3f}"
            if "cld_max" in r:
                row_str += f"  CLD={r['cld_max']:.8f}"
            if "fh_pct" in r:
                row_str += f"  FH={r['fh_pct']:.1f}%"
            if "zer" in r:
                row_str += f"  ZER={r['zer']:.3f}"
            rows.append(row_str)

        context = f"""
EVALUATION RESULTS
==================
Total experiments  : {len(self.df)}
Valid runs         : {len(df)}

PARAMETER RECOVERY
------------------
Beta  error (%)    : {stat('Beta_Error_Pct')}
Gamma error (%)    : {stat('Gamma_Error_Pct')}

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
        """
        Generates a structured report covering all metric categories
        and saves it to markdown.
        """
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

    def explain_experiment(self, experiment_id: int) -> str:
        """
        Asks GPT to explain what happened in a specific experiment —
        why the errors are what they are and what the metrics mean
        for that particular epidemic profile.
        """
        row = self.df[self.df["Experiment_ID"] == experiment_id]
        if row.empty:
            return f"Experiment {experiment_id} not found in CSV."
        return self.ask(
            f"Explain the results of this specific experiment in detail. "
            f"What do the errors and metrics tell us about this epidemic profile?\n"
            f"{row.to_string(index=False)}"
        )

    def compare_experiments(self, metric: str = "Beta_Error_Pct") -> str:
        """
        Asks GPT to explain what separates the best and worst experiments
        by the given metric column.
        """
        df = self.df.copy()
        if "Status" in df.columns:
            df = df[df["Status"] == "ok"]

        if metric not in df.columns:
            return f"Column '{metric}' not found in CSV."

        cols = ["Experiment_ID", "True_Beta", "True_Gamma", metric]
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
        """
        Generates a 3-paragraph non-technical brief suitable for
        public health ministers or hospital administrators.
        """
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