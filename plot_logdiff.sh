#!/usr/bin/env bash
set -ex
#diff_args=(--diff-vmin -10 --diff-vmax 0 --eps 1e-8)
diff_args=(--diff-vmin -10 --diff-vmax 0)
#plot_args=(--vmin 0 --vmax 0.5)

iter_final_sl=130
iter_final_ml=56
iter_ref_sl=300
levels=4


# Plots for multi-level case
for iters in 3 10 20 "$iter_final_ml"; do
    if [[ "$iters" == "$iter_final_ml" ]]; then
        label=final
    else
        label=it"$iters"
    fi

    python3 ../plot_solution.py \
        --reference         solution_2d_sl_b1000_lvl1_iter"$iter_ref_sl".bin \
        --reference-coords  solution_2d_sl_b1000_lvl1_coords.bin \
        solution_2d_ml_b1000_lvl"$levels"_coords.bin \
        solution_2d_ml_b1000_lvl"$levels"_iter"$iters".bin \
        "${diff_args[@]}" --no-colorbar --title ""

    mv solution.png         gs_l"$levels"_"$label"_sol.png
    mv solution_logdiff.png gs_l"$levels"_"$label".png
done


# Plots for single-level case
for iters in 3 10 20 "$iter_final_sl"; do
    if [[ "$iters" == "$iter_final_sl" ]]; then
        label=final
    else
        label=it"$iters"
    fi

    python3 ../plot_solution.py \
        --reference         solution_2d_sl_b1000_lvl1_iter"$iter_ref_sl".bin \
        --reference-coords  solution_2d_sl_b1000_lvl1_coords.bin \
        solution_2d_sl_b1000_lvl1_coords.bin \
        solution_2d_sl_b1000_lvl1_iter"$iters".bin \
        "${diff_args[@]}" --no-colorbar --title ""

    mv solution.png         gs_l1_"$label"_sol.png
    mv solution_logdiff.png gs_l1_"$label".png
done
