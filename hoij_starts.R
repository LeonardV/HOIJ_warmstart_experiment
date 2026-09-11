# =====================================================================
# hoij_starts.R -- IJ1- en HOIJ-2-startwaarden voor bootstrap-resamples
#
# Gedeeld door HOJ_warmstart.R (het experiment) en optimizer_experiments.R.
# Vereist dat hoij_core.R (kernel uit HOIJ-lavaan) al gesourcet is.
# =====================================================================

if (!exists("ij1_replicates", mode = "function"))
  source(file.path(Sys.getenv("HOIJ_LAVAAN_DIR", "."), "hoij_core.R"))

# ═══════════════════════════════════════════════════════════════════════════════
# HOIJ STARTWAARDEN                                                        [W19c]
#
# Wrapper rond de kernel: voor een B x N indexmatrix worden de IJ1- en (mits
# H_all en T_arr aanwezig) de HOIJ-2-startwaarden voor alle resamples in één
# keer berekend.
#
# [W7] Trust-region damping: de 2e-orde-correctie s2 = HOIJ-2 - IJ1
#      (= Bc - Ac in de notatie van de kernel) wordt afgekapt op
#      kappa_damp × ||c_vec||. Volle multinomiale gewichten kunnen buiten de
#      convergentiestraal van de expansie vallen; een ontspoorde 2e-orde-stap
#      is dan slechter dan IJ1. De ongedempte ratio ||s2||/||c|| wordt
#      teruggegeven als diagnostiek.
# ═══════════════════════════════════════════════════════════════════════════════

idx_to_dW <- function(idx_mat, N) {
  # bootstrap-frequenties - 1, één rij per resample (B x N)
  dW <- t(apply(idx_mat, 1, tabulate, nbins = N)) - 1L
  matrix(dW, nrow = nrow(idx_mat), ncol = N)
}

hoij_starts <- function(fit, theta0, Scores, H.inv, idx_mat,
                        H_all = NULL, T_arr = NULL, kappa_damp = 1) {
  N  <- nrow(fit@Data@X[[1]])
  dW <- idx_to_dW(idx_mat, N)
  stopifnot(all(rowSums(dW) == 0L))

  ij1 <- ij1_replicates(theta0, Scores, H.inv, dW)            # theta0 - c
  out <- list(ij1 = ij1$theta, C = ij1$C)

  if (!is.null(H_all) && !is.null(T_arr)) {
    th2 <- hoij2_replicates(theta0, ij1$C, dW, H.inv, H_all, T_arr,
                            fit = fit)                        # theta0 - c + (Bc - Ac)
    s2  <- th2 - ij1$theta
    n1  <- sqrt(rowSums(ij1$C^2))
    n2  <- sqrt(rowSums(s2^2))
    damped <- is.finite(kappa_damp) & n2 > kappa_damp * n1    # [W7]
    scale  <- ifelse(damped, kappa_damp * n1 / n2, 1)
    out$ij2      <- ij1$theta + s2 * scale
    out$s2_ratio <- n2 / n1
    out$damped   <- damped
  }
  out
}
