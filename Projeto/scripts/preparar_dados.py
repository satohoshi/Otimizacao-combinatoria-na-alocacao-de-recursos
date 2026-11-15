"""gera mu e sigma pra uma data t e exporta a instancia pro c++

uso:
    python3 scripts/preparar_dados.py                 # t = 2014-01-02, yahoo + b3
    python3 scripts/preparar_dados.py --t 2015-01-02
    python3 scripts/preparar_dados.py --sem-b3        # so yahoo

saidas (<tag> = <t>_b3 ou <t>_semb3):
    data/processado/<tag>/   mu.csv, sigma_amostral.csv, sigma_lw.csv,
                             exclusoes.csv, eventos_extremos.csv, pares_correlacionados.csv,
                             resumo.json
    data/instancias/<tag>/amostral/  e  .../lw/   assets.csv + covariance.csv (formato c++)
    data/processado/eventos_extremos_periodo_todo_<b3|semb3>.csv   (todos de 2008-2026, so pra conferir)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dados  # noqa: E402

RAIZ = dados.RAIZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="2014-01-02", help="data de decisão (usa só dados < t)")
    ap.add_argument("--sem-b3", action="store_true", help="só Yahoo (sem as séries do COTAHIST)")
    args = ap.parse_args()

    brutos = dados.carregar(usar_b3=not args.sem_b3)
    est = dados.estimar(brutos, args.t)
    fonte = "semb3" if args.sem_b3 else "b3"
    tag = f"{est.t.date().isoformat()}_{fonte}"

    out = RAIZ / "data" / "processado" / tag
    out.mkdir(parents=True, exist_ok=True)
    est.mu.rename("mu_anual").to_csv(out / "mu.csv", index_label="ticker")
    est.sigma.to_csv(out / "sigma_amostral.csv")
    est.sigma_lw.to_csv(out / "sigma_lw.csv")
    est.exclusoes.to_csv(out / "exclusoes.csv", index=False)
    est.eventos.to_csv(out / "eventos_extremos.csv", index=False)
    est.pares_correlacionados.to_csv(out / "pares_correlacionados.csv", index=False)
    dados.exportar_instancia(est, RAIZ / "data" / "instancias" / tag / "amostral")
    dados.exportar_instancia(est, RAIZ / "data" / "instancias" / tag / "lw", usar_lw=True)

    psd_a, min_a, max_a = dados.eh_psd(est.sigma)
    psd_l, min_l, max_l = dados.eh_psd(est.sigma_lw)
    vol = np.sqrt(np.diag(est.sigma))
    resumo = {
        "t": tag, "janela_inicio": est.inicio.date().isoformat(), "janela_fim": est.fim.date().isoformat(),
        "n_pregoes": int(len(est.retornos)), "n_universo": int(brutos.precos.shape[1]),
        "n_excluidos": int(len(est.exclusoes)), "N": len(est.tickers),
        "N_por_fonte": {f: sum(brutos.origem[tk] == f for tk in est.tickers) for f in ("yahoo", "b3")},
        "amostral": {"psd": psd_a, "autovalor_min": min_a, "autovalor_max": max_a, "condicionamento": max_a / min_a},
        "ledoit_wolf": {"psd": psd_l, "autovalor_min": min_l, "autovalor_max": max_l,
                        "condicionamento": max_l / min_l, "shrinkage": est.shrinkage_lw},
        "mu_anual": {"min": float(est.mu.min()), "mediana": float(est.mu.median()), "max": float(est.mu.max())},
        "vol_anual": {"min": float(vol.min()), "mediana": float(np.median(vol)), "max": float(vol.max())},
        "eventos_extremos": est.eventos.classificacao.value_counts().to_dict() if len(est.eventos) else {},
    }
    (out / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))

    # |r| > 20% no periodo todo (n entra na estimacao, so pra conferir)
    p = dados.preencher(brutos.precos)
    _, todos = dados.limpar_extremos(dados.retornos_simples(p), dados.retornos_simples(brutos.ibov),
                                     brutos.negocios)
    todos.to_csv(RAIZ / "data" / "processado" / f"eventos_extremos_periodo_todo_{fonte}.csv", index=False)

    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print("\nExclusões:\n" + est.exclusoes.to_string(index=False))
    print("\nPares correlacionados:\n" + est.pares_correlacionados.round(3).to_string(index=False))
    print("\nEventos extremos na janela:\n" + est.eventos.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
