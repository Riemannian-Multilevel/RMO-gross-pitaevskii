#!/usr/bin/env bash
set -xe
argv0=./main_coarse
out_dir=mass_consistent/

# Problem parameters ----------------------------------------------------------------------------------------------------------
#beta_range=(100 1000 10000)
beta_range=(1000)
#iter_range=(20 40 60)
iter_range=(150)
kappa=0.5
eps=2e-8
#eps=1e-2
tol_res=1e-8
line_search=1
radius=11
boundary=dirichlet
potential=optical_lattice
mass_lumping=0


# Arguments -------------------------------------------------------------------------------------------------------------------
args=(--potential "$potential" --boundary "$boundary" --radius "$radius")

if (( line_search )); then
    args+=(--line-search)
fi

if (( mass_lumping )); then
    args+=(--mass-lumping)
fi

# Coarse condition
args_fas=(--kappa "$kappa" --eps "$eps")

# Define file names
suffix=_optical_lattice.org
#suffix=_optical_lattice_lumped.org
#suffix=.org


# Multilevel hierarchy --------------------------------------------------------------------------------------------------------
top_level=11
#top_level=10

levels=()
levels[0]="$top_level,$((top_level-1))"
levels[1]="$top_level,$((top_level-1)),$((top_level-2))"
levels[2]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3))"
levels[3]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3)),$((top_level-4))"
#levels[4]="$top_level,$((top_level-1)),$((top_level-2)),$((top_level-3)),$((top_level-4)),$((top_level-5))"


# Main loop -------------------------------------------------------------------------------------------------------------------
num_exp=8
#num_exp=4
total=$((${#beta_range[@]} * ${#levels[@]} * num_exp))
count=0

for i in "${!beta_range[@]}"; do
    beta=${beta_range[$i]}
    iter=${iter_range[$i]}

    # Single level reference
    $argv0 "${args[@]}" --levels "$top_level" --metric none --max-iter 1000 --tol-residual 1e-14 --beta "$beta" \
	    >"$out_dir/sl_b${beta}_l${top_level}${suffix}"
    
    
    # Multilevel
    for depth in "${!levels[@]}"; do
        depth_lab=$((depth + 2))

        # Mass metric
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport mass \
            >"$out_dir/ml_mass_proj_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport differential \
            >"$out_dir/ml_mass_diff_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport adjoint_restriction \
            >"$out_dir/ml_mass_adj1_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric mass --transport adjoint_differential \
            >"$out_dir/ml_mass_adj2_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        

        # Euclidean metric
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric frobenius --transport frobenius \
            >"$out_dir/ml_frob_proj_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric frobenius --transport differential_frobenius \
            >"$out_dir/ml_frob_diff_b${beta}_l${top_level}_depth${depth_lab}${suffix}"

        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric frobenius --transport adjoint_restriction_frobenius \
            >"$out_dir/ml_frob_adj1_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        
        echo "[$(( ++count ))/$total]"
        $argv0 "${args[@]}" "${args_fas[@]}" --multilevel "${levels[$depth]}" --max-iter "$iter" --tol-residual "$tol_res" --beta "$beta" --metric frobenius --transport adjoint_differential_frobenius \
            >"$out_dir/ml_frob_adj2_b${beta}_l${top_level}_depth${depth_lab}${suffix}"
        
    done
done
