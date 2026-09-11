# =====================================================================
# optimizer_experiments.R
#
# Beoordeling van warme starts (theta_hat, IJ1, HOIJ-2) voor de lavaan-
# optimizers nlminb en GN, op hetzelfde bifactormodel en dezelfde data
# als HOJ_warmstart. Vier experimenten, elk op B_EXP bootstrap-resamples:
#
#   (A) iteraties per startwaarde en optimizer, met als vloer een start
#       op het exacte resample-optimum;
#   (B) iteraties als functie van de startafstand tot het resample-
#       optimum (gestoorde exacte start), om te zien of de positie of
#       iets anders (tolerantie, krommingsopbouw) de iteraties bepaalt;
#   (C) overdracht van kromming: eigenwaarden van H^-1 H_b (full-sample
#       informatie tegen resample-informatie in het resample-optimum),
#       plus een Newton-iteratie met de vaste full-sample H^-1;
#   (D) interactie tussen tolerantie (nlminb rel.tol, GN optim.gn.tol.x)
#       en startwaarde, met de afwijking van het strakke optimum op de
#       schaal van de standaardfouten.
#
# Werkdirectory = repositoryroot. Vereist hoij_core.R en hoij_starts.R.
# Looptijd: enkele minuten. Resultaten worden ook als RDS weggeschreven
# naar optimizer_experiments_output/.
# =====================================================================

suppressPackageStartupMessages(library(lavaan))
source("hoij_core.R")
source("hoij_starts.R")

B_EXP     <- 30          # aantal bootstrap-resamples per experiment
SAMPLE_N  <- 500
SEED_DATA <- 42          # zelfde seed als DATA_SEED_BASE in HOJ_warmstart
SEED_IDX  <- 4242        # zelfde seed als idx_mat in HOJ_warmstart
MEANSTRUCTURE <- TRUE
KAPPA_DAMP    <- 1

out_dir <- "optimizer_experiments_output"
if (!dir.exists(out_dir)) dir.create(out_dir)
cat("lavaan", as.character(packageVersion("lavaan")), "\n")


# ---------------------------------------------------------------------
# Model en data (identiek aan HOJ_warmstart)
# ---------------------------------------------------------------------
pop.syntax <- '
  g =~ 0.7*y1 + 0.6*y2 + 0.5*y3 + 0.55*y4 +
       0.65*y5 + 0.6*y6 + 0.5*y7 + 0.45*y8 +
       0.6*y9 + 0.55*y10 + 0.5*y11 + 0.45*y12
  s1 =~ 0.4*y1 + 0.35*y2 + 0.3*y3 + 0.25*y4
  s2 =~ 0.35*y5 + 0.4*y6 + 0.3*y7 + 0.25*y8
  s3 =~ 0.3*y9 + 0.35*y10 + 0.4*y11 + 0.25*y12
  g ~~ 0*s1 + 0*s2 + 0*s3
  s1 ~~ 0*s2 + 0*s3
  s2 ~~ 0*s3
  g ~~ 1*g
  s1 ~~ 1*s1
  s2 ~~ 1*s2
  s3 ~~ 1*s3
'
syntax <- '
  g =~ y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12
  s1 =~ y1 + y2 + y3 + y4
  s2 =~ y5 + y6 + y7 + y8
  s3 =~ y9 + y10 + y11 + y12
  g ~~ 0*s1 + 0*s2 + 0*s3
  s1 ~~ 0*s2 + 0*s3
  s2 ~~ 0*s3
'

set.seed(SEED_DATA)
DATA <- simulateData(pop.syntax, sample.nobs = SAMPLE_N, std.lv = TRUE)
N <- nrow(DATA)

# Eén fitfunctie: NULL bij een fout, verder zoals fit_boot in HOJ_warmstart,
# maar met de optimizer als argument en extra lavaan-opties via ...
fitf <- function(dat, method = "nlminb", start = NULL, ...) {
  suppressWarnings(tryCatch(
    if (is.null(start)) {
      sem(syntax, data = dat, std.lv = TRUE, estimator = "ML",
          meanstructure = MEANSTRUCTURE, optim.method = method,
          se = "none", ...)
    } else {
      sem(syntax, data = dat, std.lv = TRUE, estimator = "ML",
          meanstructure = MEANSTRUCTURE, optim.method = method,
          se = "none", start = start, ...)
    },
    error = function(e) NULL))
}
conv_of <- function(f) !is.null(f) && isTRUE(lavInspect(f, "converged"))
adm_of  <- function(f) conv_of(f) &&
  isTRUE(tryCatch(lavInspect(f, "post.check"), error = function(e) FALSE))
iter_of <- function(f) if (is.null(f)) NA_real_ else
  as.numeric(lavInspect(f, "optim")$iterations)
fx_of   <- function(f) if (is.null(f)) NA_real_ else
  as.numeric(lavInspect(f, "optim")$fx)


# ---------------------------------------------------------------------
# Referentiefit en HOIJ-setup (zoals in HOJ_warmstart)
# ---------------------------------------------------------------------
fit <- fitf(DATA, "nlminb")
stopifnot(adm_of(fit))
theta_hat <- coef(fit, type = "free")
D <- length(theta_hat)
cat(sprintf("N = %d, D = %d, meanstructure = %s\n", N, D, MEANSTRUCTURE))

Scores <- lavScores(fit, scaling = TRUE)
H.inv  <- lavTech(fit, "inverted.information.observed")
H_obs  <- lavTech(fit, "information.observed")
grad_F <- make_grad_F(fit)
chk <- check_gradient_hessian(grad_F, theta_hat, H_obs)
stopifnot(is.finite(chk$spread), chk$spread < 0.1)
t_setup <- system.time({
  H_all <- compute_all_H(fit, theta_hat)
  T_arr <- compute_T_tensor_grad(grad_F, theta_hat)
})[["elapsed"]]
cat(sprintf("HOIJ-2 setup: %.2f s\n", t_setup))

set.seed(SEED_IDX)
idx_mat <- matrix(sample.int(N, B_EXP * N, replace = TRUE), nrow = B_EXP)
st_damped   <- hoij_starts(fit, theta_hat, Scores, H.inv, idx_mat,
                           H_all, T_arr, kappa_damp = KAPPA_DAMP)
st_undamped <- hoij_starts(fit, theta_hat, Scores, H.inv, idx_mat,
                           H_all, T_arr, kappa_damp = Inf)

# Strak resample-optimum per resample (nlminb, default tolerantie) als
# referentie voor afstanden en als "exacte" start.
cat("Exacte resample-optima (nlminb)...\n")
exact <- lapply(seq_len(B_EXP), function(b) {
  f <- fitf(DATA[idx_mat[b, ], ], "nlminb")
  list(fit = f, conv = conv_of(f), adm = adm_of(f),
       theta = if (conv_of(f)) as.numeric(coef(f, type = "free")) else NULL)
})
adm_b <- which(vapply(exact, `[[`, logical(1), "adm"))
cat(sprintf("  %d/%d resamples met een toelaatbaar (post.check) optimum\n",
            length(adm_b), B_EXP))

starts_of <- function(b) list(
  default        = NULL,
  theta_hat      = as.numeric(theta_hat),
  ij1            = as.numeric(st_damped$ij1[b, ]),
  hoij2          = as.numeric(st_damped$ij2[b, ]),
  hoij2_undamped = as.numeric(st_undamped$ij2[b, ]),
  exact          = exact[[b]]$theta)
start_levels <- c("default", "theta_hat", "ij1", "hoij2", "hoij2_undamped",
                  "exact")

dist_to <- function(start, b, f) {
  th_b <- exact[[b]]$theta
  if (is.null(th_b)) return(NA_real_)
  if (is.null(start)) {                       # lavaan-default start
    if (is.null(f)) return(NA_real_)
    pt <- parTable(f); start <- pt$start[pt$free > 0]
  }
  sqrt(sum((start - th_b)^2))
}

# Mediaan over de resamples waarin alle armen convergeerden én toelaatbaar
# waren én naar hetzelfde optimum gingen (fx-spreiding <= 1e-6), zoals
# [W13]/[W8] in HOJ_warmstart.
same_opt_subset <- function(r) {
  ok_b <- as.integer(names(which(tapply(r$adm, r$b, all))))
  w <- reshape(r[, c("b", "arm", "fx")], idvar = "b", timevar = "arm",
               direction = "wide")
  fxm <- as.matrix(w[, -1])
  spread <- (apply(fxm, 1, max) - apply(fxm, 1, min)) / abs(apply(fxm, 1, min))
  intersect(ok_b, w$b[is.finite(spread) & spread <= 1e-6])
}


# =====================================================================
# (A) iteraties per startwaarde en optimizer
# =====================================================================
cat("\n(A) startwaarden x optimizer...\n")
resA <- list()
for (b in seq_len(B_EXP)) {
  dat_b <- DATA[idx_mat[b, ], ]
  starts <- starts_of(b)
  for (m in c("nlminb", "GN")) for (s in start_levels) {
    if (is.null(starts[[s]]) && s != "default") next
    tt <- system.time(f <- fitf(dat_b, m, starts[[s]]))[["elapsed"]]
    resA[[length(resA) + 1]] <- data.frame(
      b = b, method = m, arm = s, conv = conv_of(f), adm = adm_of(f),
      iter = iter_of(f), fx = fx_of(f), time = tt,
      dist = dist_to(starts[[s]], b, f))
  }
}
resA <- do.call(rbind, resA)
resA$arm <- factor(resA$arm, levels = start_levels)
for (m in c("nlminb", "GN")) {
  r <- resA[resA$method == m, ]
  sel <- same_opt_subset(r)
  cat(sprintf("\n== (A) %s: mediaan over %d resamples (alle armen toelaatbaar, zelfde optimum) ==\n",
              m, length(sel)))
  tab <- do.call(rbind, lapply(split(r, r$arm), function(d) data.frame(
    start = d$arm[1], n_conv = sum(d$conv), n_adm = sum(d$adm),
    dist_med = median(d$dist, na.rm = TRUE),
    iter_med = median(d$iter[d$b %in% sel]),
    iter_mean = mean(d$iter[d$b %in% sel]),
    time_ms = 1000 * median(d$time[d$b %in% sel]))))
  print(tab, digits = 3, row.names = FALSE)
}


# =====================================================================
# (B) iteraties als functie van de startafstand
# =====================================================================
cat("\n(B) gestoorde exacte start (afstand eps langs een random richting)...\n")
EPS <- c(0.02, 0.1, 0.5, 1.5)
resB <- list()
for (b in adm_b) {
  dat_b <- DATA[idx_mat[b, ], ]; th_b <- exact[[b]]$theta
  set.seed(b); u <- rnorm(D); u <- u / sqrt(sum(u^2))
  for (eps in EPS) for (m in c("nlminb", "GN")) {
    f <- fitf(dat_b, m, th_b + eps * u)
    resB[[length(resB) + 1]] <- data.frame(b = b, eps = eps, method = m,
                                           conv = conv_of(f), iter = iter_of(f))
  }
}
resB <- do.call(rbind, resB)
tabB <- do.call(rbind, lapply(split(resB, list(resB$method, resB$eps)),
  function(d) data.frame(method = d$method[1], eps = d$eps[1], n = nrow(d),
                         n_conv = sum(d$conv),
                         iter_med = median(d$iter[d$conv]),
                         iter_max = max(d$iter[d$conv]))))
print(tabB[order(tabB$method, tabB$eps), ], row.names = FALSE)


# =====================================================================
# (C) overdracht van kromming en een Newton-iteratie met vaste H^-1
# =====================================================================
cat("\n(C) kromming: eigenwaarden van H^-1 H_b; Newton met vaste full-sample H^-1...\n")
ev <- eigen(H_obs, symmetric = TRUE, only.values = TRUE)$values
cat(sprintf("  H_obs in theta_hat: eigenwaarden [%.3g, %.3g], conditiegetal %.3g\n",
            min(ev), max(ev), max(ev) / min(ev)))
resC <- list(); chord <- list()
for (b in seq_len(B_EXP)) {
  if (!exact[[b]]$conv) next
  Hb  <- lavTech(exact[[b]]$fit, "information.observed")
  lam <- Re(eigen(H.inv %*% Hb, only.values = TRUE)$values)
  evb <- eigen(Hb, symmetric = TRUE, only.values = TRUE)$values
  resC[[length(resC) + 1]] <- data.frame(
    b = b, admissible = exact[[b]]$adm, lam_min = min(lam), lam_max = max(lam),
    n_outside_0_2 = sum(lam <= 0 | lam >= 2), cond_Hb = max(evb) / min(evb))
  if (!exact[[b]]$adm) next
  ## Newton-iteratie x <- x - H^-1 grad_b(x) met de exacte resample-gradient
  ## (convergeert alleen als alle eigenwaarden van H^-1 H_b in (0, 2) liggen)
  f0 <- fitf(DATA[idx_mat[b, ], ], "nlminb", do.fit = FALSE)
  gb <- make_grad_F(f0); th_b <- exact[[b]]$theta
  for (s in c("theta_hat", "ij1", "hoij2")) {
    x <- starts_of(b)[[s]]; d_path <- sqrt(sum((x - th_b)^2)); k <- 0
    repeat {
      k <- k + 1; step <- drop(H.inv %*% gb(x)); x <- x - step
      d_path <- c(d_path, sqrt(sum((x - th_b)^2)))
      if (sqrt(sum(step^2)) < 1e-6 || k >= 50 || !all(is.finite(x))) break
    }
    chord[[length(chord) + 1]] <- data.frame(
      b = b, start = s, iters = k, converged = sqrt(sum(step^2)) < 1e-6,
      d0 = d_path[1], d_after_1 = d_path[2], d_final = tail(d_path, 1))
  }
}
resC <- do.call(rbind, resC); chord <- do.call(rbind, chord)
cat("  eigenwaarden van H^-1 H_b per resample:\n")
print(resC, digits = 3, row.names = FALSE)
cat("  Newton met vaste H^-1 (toelaatbare resamples):\n")
print(do.call(rbind, lapply(split(chord, chord$start), function(d) data.frame(
  start = d$start[1], n = nrow(d), n_converged = sum(d$converged),
  d0_med = median(d$d0), d_after_1_med = median(d$d_after_1),
  d_final_med = median(d$d_final)))), digits = 3, row.names = FALSE)


# =====================================================================
# (D) tolerantie x startwaarde
# =====================================================================
cat("\n(D) tolerantie x startwaarde...\n")
REL_TOL <- c(1e-10, 1e-8, 1e-6)      # nlminb rel.tol (lavaan-default 1e-10)
GN_TOL  <- c(1e-5, 1e-4, 1e-3)       # optim.gn.tol.x (lavaan-default 1e-5)
se_norm <- sqrt(sum(diag(vcov(sem(syntax, data = DATA, std.lv = TRUE,
                                  meanstructure = MEANSTRUCTURE)))))
resD <- list()
for (b in adm_b) {
  dat_b <- DATA[idx_mat[b, ], ]; th_b <- exact[[b]]$theta
  starts <- starts_of(b)
  for (s in c("default", "theta_hat", "ij1", "hoij2")) {
    for (rt in REL_TOL) {
      tt <- system.time(f <- fitf(dat_b, "nlminb", starts[[s]],
                                  control = list(rel.tol = rt)))[["elapsed"]]
      resD[[length(resD) + 1]] <- data.frame(
        b = b, method = "nlminb", tol = rt, arm = s, conv = conv_of(f),
        iter = iter_of(f), time = tt,
        err = if (conv_of(f)) sqrt(sum((coef(f, type = "free") - th_b)^2)) else NA)
    }
    for (tx in GN_TOL) {
      tt <- system.time(f <- fitf(dat_b, "GN", starts[[s]],
                                  optim.gn.tol.x = tx))[["elapsed"]]
      resD[[length(resD) + 1]] <- data.frame(
        b = b, method = "GN", tol = tx, arm = s, conv = conv_of(f),
        iter = iter_of(f), time = tt,
        err = if (conv_of(f)) sqrt(sum((coef(f, type = "free") - th_b)^2)) else NA)
    }
  }
}
resD <- do.call(rbind, resD)
resD$arm <- factor(resD$arm, levels = start_levels[1:4])
cat(sprintf("  norm van de standaardfouten over %d parameters: %.3f\n", D, se_norm))
for (m in c("nlminb", "GN")) {
  r <- resD[resD$method == m, ]
  ok_b <- as.integer(names(which(tapply(r$conv, r$b, all))))
  r <- r[r$b %in% ok_b, ]
  cat(sprintf("\n== (D) %s: mediaan over %d resamples; err = afstand tot strak optimum ==\n",
              m, length(ok_b)))
  tab <- aggregate(cbind(iter, time, err) ~ tol + arm, data = r, FUN = median)
  tab$time_ms <- round(1000 * tab$time, 1); tab$time <- NULL
  tab$err_rel_se <- tab$err / se_norm
  print(tab[order(tab$tol, tab$arm), ], digits = 3, row.names = FALSE)
}

saveRDS(list(A = resA, B = resB, C = list(eig = resC, chord = chord), D = resD,
             lavaan = as.character(packageVersion("lavaan")),
             sessionInfo = sessionInfo()),
        file.path(out_dir, sprintf("optimizer_experiments_%s.rds",
                                   format(Sys.time(), "%Y%m%d_%H%M"))))
cat(sprintf("\nKlaar. RDS in %s/\n", out_dir))
