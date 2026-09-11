# Higher-Order Infinitesimal Jackknife (Giordano, Jordan & Broderick, 2019)
# Approximeert θ̂(w) via Taylor-expansie in de gewichten
#
# Warm-start-experiment: HOIJ-startwaarden vs IJ1 vs default vs theta_hat
# voor een bifactormodel met 12 indicatoren (N = 500, B = 500).
#
# De HOIJ-berekening zelf (casewise loglik, casewise informatie H_i,
# derde-afgeleide-array, IJ1- en HOIJ-2-replicaten) komt uit hoij_core.R van
# de HOIJ-lavaan-repository (Vanbrabant & Rosseel). Dit script bevat alleen
# nog de experiment-specifieke laag: pre-flight checks, trust-region
# damping, ridge-correctie, de bootstrap-loop met vijf armen en de
# rapportage. Werkdirectory = repositoryroot (hoij_core.R wordt daar
# gesourcet), conform de conventie in HOIJ-lavaan.
#
# Vereisten: lavaan (development-versie, zie 00_install_dependencies.R in
# HOIJ-lavaan), glmnet, MASS.
#
# ─── CHANGELOG v2 ────────────────────────────────────────────────────────────
# [W1]  Onvoorwaardelijke install.packages() verwijderd (herinstalleerde lavaan
#       bij elke run en brak de versie-pin). Vervangen door versie-check met
#       melding. requireNamespace-guard voor MASS toegevoegd (ginv-fallback).
# [W2]  Bootstrap-indices pregegenereerd in idx_mat (aparte seed, vóór de loop)
#       zodat cv.glmnet-RNG de resamples niet kan beïnvloeden. Consistent met
#       het gedeelde-indexmatrix-patroon in de benchmarkscript.
# [W3]  Pre-flight checks toegevoegd:
#       (a) orde-1 teken/schaal-validatie: 3 validatie-resamples, IJ1-start
#           moet dichter bij het exacte refit-optimum liggen dan theta_hat;
#       (b) B-term-schaalassert: apply(J_all, c(2,3), mean) moet H_obs
#           reproduceren (unit observed information). Vangt stille
#           deler-/factorfouten in de casewise-loglik-route af.
# [W4]  calibrate_alpha: Richardson-extrapolatie op de FD-Hessiaan (ruis-
#       reductie i.p.v. tolerantieverruiming); h 1e-5 → 1e-4; tol 0.1 → 0.01;
#       gemeten spread wordt altijd gerapporteerd.
# [W5]  Stapgroottes geharmoniseerd: compute_all_J delta 1e-5 → 1e-4
#       (eps^(1/4)-regel voor tweede differenties); ll_0 wordt eenmalig
#       berekend i.p.v. D keer in de diagonaallus.
# [W6]  IJ1-arm toegevoegd (start, fit, iteraties, afstand). Zonder deze arm
#       is winst niet attribueerbaar aan het 2e-orde-mechaniek. Afstands-
#       vectoren geïnitialiseerd op NA i.p.v. 0.
# [W7]  Trust-region damping in theta_hoij (kappa_damp, default 1): de
#       2e-orde-correctie wordt afgekapt op kappa_damp × ||c_vec||;
#       s2/c-ratio wordt per replicatie gelogd (diagnostiek convergentiestraal).
# [W8]  fx (fitfunctiewaarde bij convergentie) per arm opgeslagen;
#       zelfde-optimum-check na afloop (lokale-minima-diagnostiek).
# [W9]  Iteraties via lavInspect(fit, "optim")$iterations i.p.v. interne slot
#       @Fit@iterations.
# [W10] Alle lavaan-fits gebruiken expliciet optim.method = "GN". Dit is een
#       rechtstreeks lavOptions-/sem-argument en hoort niet in control.
# [W11] Ridge: hertraining elke 50 replicaties i.p.v. elke iteratie (426
#       cv.glmnet-fits gereduceerd tot 9); drop() op de mgaussian-voorspelling
#       (was impliciete array-recycling); vóór het eerste ridge-model wordt
#       fit_ij2 hergebruikt i.p.v. een identieke dubbele fit.
# [W12] Setup-kostenboekhouding: timing + telling van gradient- en loglik-
#       evaluaties in de setup, naast het aantal door GN bespaarde iteraties.
#       Iteraties en evaluaties worden niet één-op-één omgerekend.
# [W13] Iteratievergelijking primair op de doorsnede van replicaties waarin
#       ALLE armen convergeerden (voorkomt selectiebias door ongelijke
#       convergentiepercentages).
# [W14] Changelog-header toegevoegd conform scriptconventie.
# [W15] Convergentie in de hoofdloop vereist óók lavInspect(fit,"post.check")
#       (toelaatbare oplossing), niet alleen "converged". Bij dit bifactor-
#       model (N=150, zwakke specifieke factoren) convergeert de "exacte"
#       refit soms naar een Heywood-geval/grenswaarde; zonder deze filter
#       telt zo'n ontaarde fit als snelle, gezonde convergentie mee en
#       vervuilt hij zowel de iteratie- als de afstandsvergelijking.
#       [W3a-fix] de bijbehorende pre-flight-diagnostiek nam post.check
#       aanvankelijk niet mee in de ontaard-detectie (alleen een afstands-
#       drempel), waardoor een aantoonbaar ontaarde validatie-refit
#       (post.check = FALSE) tóch de "echte teken-/schaalfout"-stop
#       activeerde. Hersteld: isFALSE(admissible) of NA telt nu direct
#       als ontaard, ongeacht de afstand.
# [W16] Admissibiliteitscheck op de REFERENTIEFIT (theta_hat) zelf, met
#       automatische re-seed (max 20 pogingen). De [W3a]-diagnostiek liet
#       zien dat de basisafstand (~633-640) constant was over resamples
#       heen, ook wanneer de resample-fit zelf volkomen gezond was
#       (admissible=TRUE, par-range normaal) — dat wijst op een uitschieter
#       in theta_hat zelf, niet in de refits. Bij dit bifactormodel (zwakke
#       specifieke factoren, N=150) is een Heywood-geval bij de
#       oorspronkelijke trekking niet zeldzaam. Als theta_hat zelf een
#       grensoplossing is, is H.inv mogelijk (bijna-)singulier en is de hele
#       HOIJ-Taylorexpansie rond een ontaard punt theoretisch niet gedekt —
#       dit moet dus vóór de rest van de setup worden afgevangen, niet pas
#       zichtbaar worden via een verwarrende teken-/schaalfout-melding.
# [W3a-v2] De harde eis "IJ1 wint op elke validatie-resample" bleek te
#       streng: op een volkomen gezonde, toelaatbare resample (relatieve
#       stap ||c_vec||/||theta_hat|| ≈ 46%) overschoot de eerste-orde-
#       benadering incidenteel, wat normaal gedrag is bij zo'n stapgrootte
#       en geen tekenfout. Vervangen door een aggregerende toets op N_VAL=20
#       validatie-resamples: (1) gemiddelde cosinus-gelijkenis tussen de
#       IJ1-correctierichting en de werkelijke verplaatsing (robuust tegen
#       overschieten van individuele resamples; negatief gemiddelde = echte
#       tekenfout, harde stop), en (2) win rate op afstand als secundaire,
#       niet-fatale waarschuwing (< 50% ondanks positieve richting wijst op
#       een te grote stap t.o.v. de lokale kromming, geen conventiefout).
# [W17] N=500 (was 150): bij N=150 was zelfs de referentiefit vaak al
#       ontaard (6 van de 6 seeds vóór een toelaatbare fit; zie [W16]-log).
# [W18] Compatibiliteit met lavaan development (0.7-1, GitHub master):
#       lav_model_gradient → lav_model_grad, GLIST → glist; versie-bewuste
#       binding LAV_GRAD_FUN; versie-pin uitgebreid; "→" in pop.syntax
#       vervangen door "->" (dev-parser doet een unicode-gsub op de syntax).
# [W19] Migratie naar hoij_core.R uit de HOIJ-lavaan-repository. De eigen
#       kopieën van de kernelfuncties zijn verwijderd en vervangen door de
#       kernel die ook de artikelscripts en hoij_lavaan() gebruiken:
#       (a) compute_loglik_casewise, compute_all_J → compute_all_H,
#           make_grad_F en compute_T_tensor_grad komen uit hoij_core.R.
#           De versie-bewuste binding van de lavaan-internals ([W18]) zit
#           nu in .hoij_internals() (glist/GLIST wordt op naam opgelost);
#           LAV_GRAD_FUN en de versie-pin LAV_VERSIONS_OK [W1] vervallen.
#           In plaats daarvan is hoij_selftest(meanstructure = TRUE) de
#           poortwachter: de conventies (a)-(e) uit de kernelheader moeten
#           slagen vóór het experiment start.
#       (b) calibrate_alpha [W4] → check_gradient_hessian uit de kernel
#           (tolerantie spread <= 0.1, zoals in hoij_lavaan() en
#           02_timing_comparison.R). De alpha-schaling van de T-array
#           vervalt: de kernel neemt alpha = 1 (T = -Khat), wat [W18d] al
#           bevestigde. De stapgrootte van compute_all_H volgt de kernel
#           (delta = 1e-5); [W5] is daarmee niet langer van toepassing.
#       (c) theta_hoij → hoij_starts(): wrapper rond ij1_replicates() en
#           hoij2_replicates() uit de kernel. De startwaarden worden nu
#           gevectoriseerd voor alle B resamples tegelijk berekend (B x N
#           gewichtsmatrix dW, zoals in de artikelscripts). De trust-region
#           damping [W7] werkt op s2 = HOIJ-2 - IJ1 en blijft ongewijzigd.
#       (d) Alle sem()-aanroepen fitten met meanstructure = TRUE en
#           expliciet estimator = "ML", zoals de artikelscripts (gezamen-
#           lijke mean/covariantie-run). Het bifactormodel krijgt daarmee
#           12 vrije intercepten: D = 24 ladingen + 12 residuvarianties +
#           12 intercepten = 48. De profiled-mean-correctie in
#           hoij2_replicates() is dan niet nodig en wordt door de kernel
#           automatisch overgeslagen.
#       (e) [W12]-tellingen volgen nu uit de kernel-formules:
#           compute_all_H = 2D^2 + 1 loglik-evaluaties;
#           check_gradient_hessian = 2D en compute_T_tensor_grad = 2D^2 + 1
#           gradientevaluaties.
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# AFHANKELIJKHEDEN EN KERNEL                                                [W19]
# ═══════════════════════════════════════════════════════════════════════════════

if (!requireNamespace("lavaan", quietly = TRUE))
  stop(paste0("lavaan ontbreekt. Installeer de development-versie, bijv. via ",
              "00_install_dependencies.R uit HOIJ-lavaan: ",
              "remotes::install_github(\"yrosseel/lavaan\", upgrade = \"never\")."))
if (!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
if (!requireNamespace("MASS",   quietly = TRUE)) install.packages("MASS")   # [W1]
suppressPackageStartupMessages({
  library(lavaan)
  library(glmnet)
})

HOIJ_CORE <- "hoij_core.R"
if (!file.exists(HOIJ_CORE))
  stop(sprintf(paste0("[W19] %s niet gevonden. Draai dit script vanuit de ",
                      "repositoryroot; hoij_core.R is de kernel uit HOIJ-lavaan."),
               HOIJ_CORE))
source(HOIJ_CORE)

cat(sprintf("lavaan versie: %s\n", as.character(packageVersion("lavaan"))))

# [W19a] Poortwachter: de lavaan-conventies waarop de kernel steunt
#        (lavScores-schaal, observed information, gradientroute, derde
#        afgeleiden, expected-information-schaal) moeten kloppen voor deze
#        lavaan-versie. Vervangt de versie-pin [W1] en de LAV_GRAD_FUN-
#        binding [W18].
if (!hoij_selftest(meanstructure = TRUE))
  stop("[W19a] hoij_core self-test gefaald; zie de melding hierboven.")


# ═══════════════════════════════════════════════════════════════════════════════
# HOIJ STARTWAARDEN                                                        [W19c]
# idx_to_dW() en hoij_starts() staan in hoij_starts.R (gedeeld met
# optimizer_experiments.R). Zie dat bestand voor de damping [W7].
# ═══════════════════════════════════════════════════════════════════════════════
source("hoij_starts.R")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATIE: HOIJ-STARTWAARDEN vs IJ1 vs DEFAULT vs THETA_HAT
#
# Model: BIFACTOR met 12 indicatoren (1 generaal + 3 specifiek, orthogonaal)
# Dit model heeft:
#   - 48 vrije parameters (24 ladingen, 12 residuvarianties, 12 intercepten)
#     bij std.lv = TRUE en meanstructure = TRUE                        [W19d]
#   - Trage convergentie (vaak 50-200 iteraties)
#   - Gevoelig voor startwaarden (Heywood cases, lokale minima)
#   - [W17] N=500 (was 150): bij N=150 was zelfs de referentiefit vaak al
#     ontaard (6 van de 6 seeds vóór een toelaatbare fit; zie [W16]-log).
#     N=500 ligt verder van de identificatiegrens en zou het aandeel
#     Heywood-gevallen sterk moeten reduceren, terwijl de zwakke specifieke
#     factoren het model nog steeds traag/startwaarde-gevoelig houden.
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_N <- 500                                                       # [W17]

# Populatiemodel: bifactor met zwakke specifieke factoren
pop.syntax <- '
  # Generaal factor (laadt op alle 12 indicatoren)
  g =~ 0.7*y1 + 0.6*y2 + 0.5*y3 + 0.55*y4 +
       0.65*y5 + 0.6*y6 + 0.5*y7 + 0.45*y8 +
       0.6*y9 + 0.55*y10 + 0.5*y11 + 0.45*y12

  # Specifieke factoren (orthogonaal, zwakke ladingen -> moeilijk te scheiden)
  # [W18e] ASCII-pijl: de dev-parser (lav_parse_model_string_open) doet een
  #        unicode-gsub op de syntaxstring en faalt op niet-ASCII-tekens in
  #        niet-UTF-8-locales (bijv. C-locale op servers/CI).
  s1 =~ 0.4*y1 + 0.35*y2 + 0.3*y3 + 0.25*y4
  s2 =~ 0.35*y5 + 0.4*y6 + 0.3*y7 + 0.25*y8
  s3 =~ 0.3*y9 + 0.35*y10 + 0.4*y11 + 0.25*y12

  # Orthogonaliteit
  g ~~ 0*s1 + 0*s2 + 0*s3
  s1 ~~ 0*s2 + 0*s3
  s2 ~~ 0*s3

  # Factor varianties gefixeerd op 1
  g ~~ 1*g
  s1 ~~ 1*s1
  s2 ~~ 1*s2
  s3 ~~ 1*s3
'

# Fitmodel (wat we schatten op de bootstrap-data)
syntax <- '
  g =~ y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12
  s1 =~ y1 + y2 + y3 + y4
  s2 =~ y5 + y6 + y7 + y8
  s3 =~ y9 + y10 + y11 + y12

  g ~~ 0*s1 + 0*s2 + 0*s3
  s1 ~~ 0*s2 + 0*s3
  s2 ~~ 0*s3
'

# [W10][W19d] Eén fitfunctie voor alle armen: std.lv voor identificatie van
# het bifactormodel, meanstructure = TRUE en estimator = "ML" zoals de
# artikelscripts, optim.method = "GN". Geeft NULL bij een fout.
fit_boot <- function(dat, start = NULL) {
  tryCatch(
    if (is.null(start)) {
      sem(syntax, data = dat, std.lv = TRUE, estimator = "ML",
          meanstructure = TRUE, optim.method = "GN")
    } else {
      sem(syntax, data = dat, std.lv = TRUE, estimator = "ML",
          meanstructure = TRUE, optim.method = "GN", start = start)
    },
    error = function(e) NULL)
}

# [W16] Admissibiliteits-check op de REFERENTIEFIT zelf. Als theta_hat een
# Heywood-geval/grenswaarde is, is de hele HOIJ-machinerie gebouwd op een
# ontaard uitgangspunt: H.inv kan bijna-singulier zijn en delta_w-stappen
# vanaf een grensoplossing zijn theoretisch niet gedekt door de Taylor-
# expansie. Met SAMPLE_N=500 [W17] verwachten we dit veel minder vaak dan bij
# N=150; de re-seed-lus blijft als vangnet staan.
DATA_SEED_BASE <- 42
MAX_SEED_TRIES <- 20
seed_try <- 0
repeat {
  seed_try <- seed_try + 1
  set.seed(DATA_SEED_BASE + seed_try - 1)
  DATA <- simulateData(pop.syntax, sample.nobs = SAMPLE_N, std.lv = TRUE)  # [W17]
  fit.lav <- suppressWarnings(fit_boot(DATA))
  ok_conv <- !is.null(fit.lav) && lavInspect(fit.lav, "converged")
  ok_admis <- ok_conv && isTRUE(tryCatch(lavInspect(fit.lav, "post.check"),
                                         error = function(e) FALSE))
  if (ok_conv && ok_admis) break
  if (seed_try >= MAX_SEED_TRIES)
    stop(sprintf(paste0(
      "[W16] Geen toelaatbare referentiefit gevonden in %d pogingen. Het ",
      "populatiemodel geeft bij N=%d kennelijk structureel Heywood-gevallen ",
      "op deze fit; overweeg N verder te verhogen of de specifieke-",
      "factorladingen te versterken vóór verder onderzoek."),
      MAX_SEED_TRIES, SAMPLE_N))
  cat(sprintf(
    "[W16] seed %d gaf een ontaarde referentiefit (converged=%s, admissible=%s); volgende seed...\n",
    DATA_SEED_BASE + seed_try - 1, ok_conv, ok_admis))
}
cat(sprintf("[W16] Toelaatbare referentiefit gevonden met seed %d (%d poging(en)).\n",
            DATA_SEED_BASE + seed_try - 1, seed_try))
cat(sprintf("      par-range theta_hat = [%.3g, %.3g]\n\n", min(coef(fit.lav)), max(coef(fit.lav))))

cat(sprintf("═══ BIFACTOR MODEL: 12 indicatoren, 4 factoren, N=%d, meanstructure ═══\n", SAMPLE_N))  # [W17][W19d]
cat("    Dit model convergeert traag en is gevoelig voor startwaarden.\n\n")

cat(sprintf("Model geconvergeerd: %s | Iteraties origineel: %d\n",
            lavInspect(fit.lav, "converged"),
            lavInspect(fit.lav, "optim")$iterations))                # [W9]

theta_hat <- coef(fit.lav, type = "free")
n_params <- length(theta_hat)
N <- nrow(DATA)
stopifnot(N == nrow(fit.lav@Data@X[[1]]), n_params == 48L)          # [W19d]
cat(sprintf("Vrije parameters: %d\n\n", n_params))

# [W2] Pregenereer ALLE resample-indices met een eigen seed, vóór de loop.
#      cv.glmnet consumeert RNG; zonder pregeneratie zou elke wijziging in de
#      ridge-timing de bootstrap-draws veranderen.
B <- 500
set.seed(4242)
N_VAL <- 20                                                            # [W3a-v2]
idx_val_mat <- matrix(sample.int(N, N_VAL * N, replace = TRUE), nrow = N_VAL)  # [W3a]
idx_mat     <- matrix(sample.int(N, B * N, replace = TRUE), nrow = B)  # [W2]

# ─── Pre-compute HOIJ-ingrediënten (eenmalig) ───
Scores <- lavScores(fit.lav, scaling = TRUE)                          # N x D
H.inv  <- lavTech(fit.lav, "inverted.information.observed")           # Hhat^-1
H_obs  <- lavTech(fit.lav, "information.observed")                    # Hhat
if (is.null(Scores) || is.null(H.inv) || is.null(H_obs))
  stop("Scores of observed information niet beschikbaar voor dit model.")

# [W19] Guard uit hoij_lavaan(): een perturbatie van theta moet de casewise
#       loglik veranderen, anders is de koppeling met de lavaan-internals
#       stilletjes verbroken.
th_pert <- theta_hat; th_pert[1] <- th_pert[1] + 1e-3
if (identical(compute_loglik_casewise(fit.lav, theta_hat),
              compute_loglik_casewise(fit.lav, th_pert)))
  stop("[W19] Koppeling met lavaan-internals verbroken: perturbatie van theta ",
       "verandert de casewise loglik niet. Zie 00_install_dependencies.R in HOIJ-lavaan.")

# ═══════════════════════════════════════════════════════════════════
# [W3a-v2] PRE-FLIGHT CHECK 1: orde-1 teken/schaal (robuuste, aggregerende toets)
#
# EERDERE VERSIE (te streng): eiste dat IJ1 op ELKE validatie-resample
# dichter bij het exacte optimum lag dan theta_hat. Dat faalde op een
# volkomen gezonde, toelaatbare resample (||c_vec||/||theta_hat|| ≈ 46%,
# d_ij1 = 1.99 vs d_hat = 1.83) — geen tekenfout, maar simpelweg een
# individuele resample met een relatief grote stap waarbij de eerste-orde
# (lineaire) benadering overschiet. Bij zo'n stapgrootte is incidenteel
# overschieten normaal gedrag van een lineaire Taylorexpansie, niet een
# signaal van een verkeerde conventie. "IJ1 wint op elke resample" is dus
# de verkeerde eis; de juiste eis is dat de correctie systematisch de
# juiste RICHTING op wijst en gemiddeld/mediaan dichterbij komt.
#
# Twee complementaire, aggregerende criteria op N_VAL niet-ontaarde
# validatie-resamples:
#   (1) Cosinus-gelijkenis tussen de IJ1-correctie (ij1_v - theta_hat) en de
#       werkelijke verplaatsing (th_v - theta_hat). Robuust tegen incidenteel
#       overschieten, want die beïnvloedt de norm van de correctie, niet de
#       richting. Gemiddelde cosinus moet duidelijk positief zijn; een
#       tekenfout in lavScores/H.inv keert de richting om en geeft een
#       gemiddelde die (ruim) negatief is.
#   (2) Win rate: aandeel resamples waarop d_ij1 < d_hat. Bij een correcte
#       conventie ruim > 50%; bij een tekenfout ruim < 50%.
# ═══════════════════════════════════════════════════════════════════
cat(sprintf("[W3a] Pre-flight: orde-1 teken/schaal-validatie (%d resamples)...\n", N_VAL))
val_starts <- hoij_starts(fit.lav, theta_hat, Scores, H.inv, idx_val_mat)  # [W19c]
ij1_val    <- val_starts$ij1                                              # N_VAL x D

c_vec_norm_ref <- sqrt(sum((ij1_val[1, ] - theta_hat)^2))
cat(sprintf("      ||c_vec|| (resample 1, sanity check) = %.4f | ||theta_hat|| = %.4f (relatieve stap %.0f%%)\n",
            c_vec_norm_ref, sqrt(sum(theta_hat^2)),
            100 * c_vec_norm_ref / sqrt(sum(theta_hat^2))))

cos_sims <- d_ij1_vec <- d_hat_vec <- rep(NA_real_, N_VAL)
n_degenerate <- 0L

for (v in 1:N_VAL) {
  idx_v <- idx_val_mat[v, ]
  ij1_v <- as.numeric(ij1_val[v, ])
  fit_v <- fit_boot(DATA[idx_v, ])
  fit_conv <- !is.null(fit_v) && lavInspect(fit_v, "converged")
  admissible <- if (fit_conv) tryCatch(lavInspect(fit_v, "post.check"),
                                       error = function(e) NA) else NA
  is_degenerate <- !fit_conv || isFALSE(admissible) || is.na(admissible)

  if (is_degenerate) {
    n_degenerate <- n_degenerate + 1L
    next
  }

  th_v  <- coef(fit_v, type = "free")
  d_ij1 <- sqrt(sum((ij1_v - th_v)^2))
  d_hat <- sqrt(sum((theta_hat - th_v)^2))
  d_ij1_vec[v] <- d_ij1
  d_hat_vec[v] <- d_hat

  move_actual <- th_v - theta_hat
  move_ij1    <- ij1_v - theta_hat
  denom <- sqrt(sum(move_actual^2)) * sqrt(sum(move_ij1^2))
  cos_sims[v] <- if (denom > 1e-10) sum(move_actual * move_ij1) / denom else NA_real_
}

n_valid   <- sum(!is.na(cos_sims))
mean_cos  <- mean(cos_sims, na.rm = TRUE)
win_rate  <- mean(d_ij1_vec < d_hat_vec, na.rm = TRUE)
cat(sprintf("      %d/%d validatie-resamples ontaard (Heywood/niet-geconvergeerd), %d bruikbaar\n",
            n_degenerate, N_VAL, n_valid))
cat(sprintf("      gemiddelde cosinus-gelijkenis (richting) = %.3f | win rate (afstand) = %.0f%% (n=%d)\n",
            mean_cos, 100 * win_rate, n_valid))

if (n_valid < 5) {
  stop(paste0("[W3a] Te weinig niet-ontaarde validatie-resamples (< 5) om de ",
              "teken-/schaalconventie te toetsen. Verhoog N_VAL of onderzoek ",
              "waarom dit bifactormodel zo vaak ontaardt (zie ook [W16])."))
}
if (mean_cos < 0) {
  stop(paste0(
    "[W3a] Gemiddelde cosinus-gelijkenis is NEGATIEF (", sprintf("%.3f", mean_cos),
    "): de IJ1-correctie wijst systematisch de VERKEERDE kant op. Dit duidt op ",
    "een echte teken-/schaalconventiefout in lavScores/H.inv. Niet doorgaan."))
}
if (win_rate < 0.5) {
  cat(sprintf(paste0(
    "[W3a] WAARSCHUWING: win rate %.0f%% < 50%% ondanks positieve gemiddelde ",
    "richting (cos=%.3f). De IJ1-stap is mogelijk te groot t.o.v. de lokale ",
    "kromming (zie relatieve stap hierboven); overweeg trust-region-damping ",
    "ook voor IJ1, of interpreteer HOIJ-2-resultaten met extra voorzichtigheid.\n"),
    100 * win_rate, mean_cos))
} else {
  cat("[W3a] OK: richting en afstand consistent met een correcte teken-/schaalconventie.\n")
}
cat("\n")

# ═══════════════════════════════════════════════════════════════════
# [W19b] Gradientroute-check vóór de dure setup (check_gradient_hessian uit
# de kernel, vervangt calibrate_alpha [W4]): de FD-Hessiaan van lavaans
# analytische gradient moet H_obs reproduceren met ratio ≈ 1 en kleine
# spread. Tolerantie 0.1 zoals in hoij_lavaan() en 02_timing_comparison.R.
# ═══════════════════════════════════════════════════════════════════
SPREAD_TOL <- 0.1
grad_F <- make_grad_F(fit.lav)
t_chk <- system.time(
  chk <- check_gradient_hessian(grad_F, theta_hat, H_obs)
)["elapsed"]
cat(sprintf("[W19b] gradient-Hessiaan-check: ratio = %.6f | spread = %.3g | tol = %.3g\n",
            chk$ratio, chk$spread, SPREAD_TOL))
if (!is.finite(chk$spread) || chk$spread > SPREAD_TOL)
  stop(sprintf(paste0(
    "[W19b] De FD-gradientroute reproduceert lavaans observed information niet ",
    "(spread %.3g > tol %.3g): de 2e-orde-stap kan voor dit model niet ",
    "betrouwbaar worden berekend. Niet doorgaan."), chk$spread, SPREAD_TOL))
if (abs(chk$ratio - 1) > 0.01)
  stop(sprintf("[W19b] Schaalratio H_obs / H_grad = %.4f wijkt af van 1.", chk$ratio))

cat("Computing per-obs Hessians H_i (bifactor, D²/2 paren)...\n")
t_H <- system.time(
  H_all <- compute_all_H(fit.lav, theta_hat)                          # [W19a] kernel-delta
)["elapsed"]

# ═══════════════════════════════════════════════════════════════════
# [W3b] PRE-FLIGHT CHECK 2: B-term-schaal
# Het gemiddelde van de casewise Hessianen moet de unit observed information
# reproduceren: mean_i H_all[i,,] ≈ H_obs. Wijkt dit af (deler N vs N-1,
# factor 2, meanstructure-conventie), dan is de B-term stilletjes fout.
# ═══════════════════════════════════════════════════════════════════
H_from_all <- apply(H_all, c(2, 3), mean)
rel_diff <- norm(H_from_all - H_obs, type = "F") / norm(H_obs, type = "F")
cat(sprintf("[W3b] Pre-flight: ||mean(H_all) - H_obs||_F / ||H_obs||_F = %.3g\n",
            rel_diff))
if (rel_diff > 1e-2)
  stop(paste0("[W3b] Casewise Hessianen reproduceren H_obs niet (rel. verschil > 1e-2): ",
              "B-term staat op een andere schaal dan H.inv. Niet doorgaan."))
cat("[W3b] OK.\n\n")

cat("Computing third-derivative array via gradient route...\n")
t_T <- system.time(
  T_arr <- compute_T_tensor_grad(grad_F, theta_hat)                   # = -Khat
)["elapsed"]

# [W12][W19e] Setup-kostenboekhouding, afgeleid uit de kernel-formules
n_ll_setup   <- 2L * n_params^2 + 1L                 # compute_all_H
n_grad_setup <- 2L * n_params + (2L * n_params^2 + 1L)  # check + T-array
cat(sprintf(paste0(
  "HOIJ setup klaar.\n[W12] Setup-kosten: %d casewise-loglik-evaluaties (H_all, %.1f s) + ",
  "%d gradientevaluaties (check %.1f s + T-array %.1f s)\n\n"),
  n_ll_setup, t_H, n_grad_setup, t_chk, t_T))

# ─── Startwaarden voor alle B resamples in één keer ───              [W19c]
KAPPA_DAMP <- 1                                                      # [W7]
t_starts <- system.time(
  starts <- hoij_starts(fit.lav, theta_hat, Scores, H.inv, idx_mat,
                        H_all = H_all, T_arr = T_arr, kappa_damp = KAPPA_DAMP)
)["elapsed"]
cat(sprintf("IJ1- en HOIJ-2-startwaarden voor B=%d resamples berekend in %.2f s\n\n",
            B, t_starts))
s2_ratio_log <- starts$s2_ratio                                      # [W7]
s2_damped    <- starts$damped

# ─── Bootstrap loop ───
RIDGE_START    <- 75
RIDGE_TRAIN_AT <- seq(RIDGE_START, B, by = 50)                       # [W11]

# Armen: default | ij1 | ij2 (gedempt) | ij2 + ridge | theta_hat      [W6]
theta_default   <- matrix(NA, B, n_params)
theta_ij2_start <- starts$ij2                # startwaarden (ridge-features)
theta_ij2r      <- matrix(NA, B, n_params)
theta_hat_mat   <- matrix(NA, B, n_params)

F_default <- F_ij1 <- F_ij2 <- F_ij2r <- F_theta_hat <- rep(NA_real_, B)
fx_default <- fx_ij1 <- fx_ij2 <- fx_ij2r <- fx_theta_hat <- rep(NA_real_, B)  # [W8]
conv_default <- conv_ij1 <- conv_ij2 <- conv_ij2r <- conv_theta_hat <- logical(B)

dist_ij1 <- dist_ij2 <- dist_ij2r <- rep(NA_real_, B)                # [W6]

ridge_model <- NULL

# [W9] Helpers voor iteraties/fx via de publieke interface
get_iter <- function(fit) {
  if (is.null(fit)) return(NA_real_)
  as.numeric(lavInspect(fit, "optim")$iterations)
}
get_fx <- function(fit) {
  if (is.null(fit)) return(NA_real_)
  as.numeric(lavInspect(fit, "optim")$fx)
}
get_admissible <- function(fit) {                                    # [W15]
  if (is.null(fit)) return(FALSE)
  isTRUE(tryCatch(lavInspect(fit, "post.check"), error = function(e) FALSE))
}

cat(sprintf("═══ Bootstrap loop: B=%d, 5 armen ═══\n", B))
t_start <- proc.time()

for (b in 1:B) {
  if (b %% 50 == 0 || b <= 5) cat(sprintf("b = %d/%d\n", b, B))
  idx   <- idx_mat[b, ]                                              # [W2]
  dat_b <- DATA[idx, ]

  # Startwaarden (vooraf berekend)                                   [W19c]
  ij1_start <- as.numeric(starts$ij1[b, ])                           # [W6]
  ij2_start <- as.numeric(starts$ij2[b, ])                           # [W7]

  # Ridge-correctie op de (gedempte) 2e-orde-start                    [W11]
  ij2r_start <- NULL
  if (!is.null(ridge_model)) {
    pred <- drop(predict(ridge_model, s = "lambda.min",
                         newx = matrix(ij2_start, nrow = 1)))        # [W11]
    ij2r_start <- as.numeric(ij2_start + pred)
  }

  fit_default <- fit_boot(dat_b)
  fit_ij1     <- fit_boot(dat_b, start = ij1_start)
  fit_ij2     <- fit_boot(dat_b, start = ij2_start)
  # [W11] vóór het eerste ridge-model is de gecorrigeerde start identiek
  #       aan ij2_start: hergebruik fit_ij2 i.p.v. een dubbele fit.
  fit_ij2r <- if (is.null(ij2r_start)) fit_ij2 else
    fit_boot(dat_b, start = ij2r_start)
  fit_theta_hat <- fit_boot(dat_b, start = as.numeric(theta_hat))

  # Opslaan resultaten (met NULL-bescherming)
  # [W15] Convergentie EN toelaatbaarheid (post.check): een fit die naar een
  # Heywood-geval/grenswaarde convergeert telt hier niet als bruikbare
  # convergentie mee, anders vervuilen ontaarde oplossingen zowel de
  # iteratie- als de afstandsvergelijking (zie [W3a]-diagnostiek).
  conv_default[b]   <- !is.null(fit_default)   && lavInspect(fit_default, "converged")   && get_admissible(fit_default)
  conv_ij1[b]       <- !is.null(fit_ij1)       && lavInspect(fit_ij1, "converged")       && get_admissible(fit_ij1)
  conv_ij2[b]       <- !is.null(fit_ij2)       && lavInspect(fit_ij2, "converged")       && get_admissible(fit_ij2)
  conv_ij2r[b]      <- !is.null(fit_ij2r)      && lavInspect(fit_ij2r, "converged")      && get_admissible(fit_ij2r)
  conv_theta_hat[b] <- !is.null(fit_theta_hat) && lavInspect(fit_theta_hat, "converged") && get_admissible(fit_theta_hat)

  F_default[b]   <- get_iter(fit_default)                            # [W9]
  F_ij1[b]       <- get_iter(fit_ij1)
  F_ij2[b]       <- get_iter(fit_ij2)
  F_ij2r[b]      <- get_iter(fit_ij2r)
  F_theta_hat[b] <- get_iter(fit_theta_hat)

  fx_default[b]   <- get_fx(fit_default)                             # [W8]
  fx_ij1[b]       <- get_fx(fit_ij1)
  fx_ij2[b]       <- get_fx(fit_ij2)
  fx_ij2r[b]      <- get_fx(fit_ij2r)
  fx_theta_hat[b] <- get_fx(fit_theta_hat)

  if (conv_default[b])   theta_default[b, ] <- coef(fit_default, type = "free")
  if (conv_ij2r[b])      theta_ij2r[b, ]    <- coef(fit_ij2r, type = "free")
  if (conv_theta_hat[b]) theta_hat_mat[b, ] <- coef(fit_theta_hat, type = "free")

  # Afstand startwaarden tot geconvergeerde (default-)oplossing        [W6]
  if (conv_default[b]) {
    dist_ij1[b]  <- sqrt(sum((theta_default[b, ] - ij1_start)^2))
    dist_ij2[b]  <- sqrt(sum((theta_default[b, ] - ij2_start)^2))
    dist_ij2r[b] <- if (is.null(ij2r_start)) dist_ij2[b] else
      sqrt(sum((theta_default[b, ] - ij2r_start)^2))
  }

  # Ridge regressie: leer systematische bias in HOIJ-benadering
  # [W11] hertraining alleen op vaste punten, niet elke iteratie
  if (b %in% RIDGE_TRAIN_AT) {
    ok <- which(conv_default[1:b] &
                  complete.cases(theta_default[1:b, , drop = FALSE]) &
                  complete.cases(theta_ij2_start[1:b, , drop = FALSE]))
    if (length(ok) >= 50) {
      x_train <- theta_ij2_start[ok, , drop = FALSE]
      y_train <- theta_default[ok, , drop = FALSE] - x_train  # residuele bias
      ridge_model <- tryCatch(
        cv.glmnet(x_train, y_train, alpha = 0, family = "mgaussian",
                  grouped = FALSE),
        error = function(e) NULL)
    }
  }
}

t_elapsed <- (proc.time() - t_start)["elapsed"]
cat(sprintf("\nBootstrap klaar in %.1f sec (%.1f min)\n", t_elapsed, t_elapsed/60))


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTATEN
# ═══════════════════════════════════════════════════════════════════════════════

cat("\n\n")
cat("═══════════════════════════════════════════════════════════════════\n")
cat(sprintf("RESULTATEN: BIFACTOR MODEL (12 items, 4 factoren, N=%d, D=%d)\n",
            SAMPLE_N, n_params))                                     # [W17][W19d]
cat("═══════════════════════════════════════════════════════════════════\n\n")

cat("─── CONVERGENTIE ───\n")
cat(sprintf("  Default:        %d/%d (%.0f%%)\n", sum(conv_default), B, 100*mean(conv_default)))
cat(sprintf("  IJ1 start:      %d/%d (%.0f%%)\n", sum(conv_ij1), B, 100*mean(conv_ij1)))
cat(sprintf("  HOIJ-2 start:   %d/%d (%.0f%%)\n", sum(conv_ij2), B, 100*mean(conv_ij2)))
cat(sprintf("  HOIJ-2 + ridge: %d/%d (%.0f%%)\n", sum(conv_ij2r), B, 100*mean(conv_ij2r)))
cat(sprintf("  Theta_hat:      %d/%d (%.0f%%)\n", sum(conv_theta_hat), B, 100*mean(conv_theta_hat)))

# ═══════════════════════════════════════════════════════════════════
# [W8] ZELFDE-OPTIMUM-CHECK
# Iteraties vergelijken is alleen zinvol als de armen naar hetzelfde
# optimum convergeren. Tel per replicatie (waar alle armen convergeerden)
# de relatieve spreiding in fx; discrepanties > 1e-6 wijzen op
# verschillende lokale minima.
# ═══════════════════════════════════════════════════════════════════
conv_all <- conv_default & conv_ij1 & conv_ij2 & conv_ij2r & conv_theta_hat  # [W13]
fx_mat <- cbind(fx_default, fx_ij1, fx_ij2, fx_ij2r, fx_theta_hat)[conv_all, , drop = FALSE]
fx_min <- apply(fx_mat, 1, min)
fx_rel_spread <- (apply(fx_mat, 1, max) - fx_min) / pmax(abs(fx_min), 1e-8)
n_discrepant <- sum(fx_rel_spread > 1e-6)
cat(sprintf("\n─── [W8] ZELFDE-OPTIMUM-CHECK (op %d replicaties waar alle armen convergeerden) ───\n",
            sum(conv_all)))
cat(sprintf("  Replicaties met fx-discrepantie > 1e-6 (rel.): %d (%.1f%%)\n",
            n_discrepant, 100 * n_discrepant / max(sum(conv_all), 1)))
if (n_discrepant > 0)
  cat("  LET OP: verschillende armen bereiken verschillende lokale optima;\n",
      " iteratievergelijking is voor die replicaties niet interpreteerbaar.\n")

# [W13] Iteraties primair op de doorsnede (en zonder fx-discrepanties)
same_opt <- rep(FALSE, B)
same_opt[which(conv_all)[fx_rel_spread <= 1e-6]] <- TRUE
iter_summary <- function(x, sel, label) {
  cat(sprintf("  %-15s mediaan=%.0f, gemiddeld=%.1f, sd=%.1f\n",
              label,
              median(x[sel], na.rm = TRUE),
              mean(x[sel], na.rm = TRUE),
              sd(x[sel], na.rm = TRUE)))
}
cat(sprintf("\n─── ITERATIES [W13: doorsnede, zelfde optimum; n=%d] ───\n", sum(same_opt)))
iter_summary(F_default,   same_opt, "Default:")
iter_summary(F_ij1,       same_opt, "IJ1:")
iter_summary(F_ij2,       same_opt, "HOIJ-2:")
iter_summary(F_ij2r,      same_opt, "HOIJ-2 + ridge:")
iter_summary(F_theta_hat, same_opt, "Theta_hat:")

# Iteratiereductie t.o.v. default, op dezelfde doorsnede
med_def  <- median(F_default[same_opt], na.rm = TRUE)
med_ij1  <- median(F_ij1[same_opt], na.rm = TRUE)
med_ij2  <- median(F_ij2[same_opt], na.rm = TRUE)
med_ij2r <- median(F_ij2r[same_opt], na.rm = TRUE)
med_hat  <- median(F_theta_hat[same_opt], na.rm = TRUE)
cat(sprintf("\n─── ITERATIEREDUCTIE t.o.v. DEFAULT (mediaan) ───\n"))
cat(sprintf("  IJ1:            -%.0f iteraties (%.0f%%)\n", med_def - med_ij1, 100*(med_def - med_ij1)/med_def))
cat(sprintf("  HOIJ-2:         -%.0f iteraties (%.0f%%)\n", med_def - med_ij2, 100*(med_def - med_ij2)/med_def))
cat(sprintf("  HOIJ-2 + ridge: -%.0f iteraties (%.0f%%)\n", med_def - med_ij2r, 100*(med_def - med_ij2r)/med_def))
cat(sprintf("  Theta_hat:      -%.0f iteraties (%.0f%%)\n", med_def - med_hat, 100*(med_def - med_hat)/med_def))

# [W7] Damping-diagnostiek
cat(sprintf("\n─── [W7] 2e-ORDE-STAP DIAGNOSTIEK ───\n"))
cat(sprintf("  ||s2||/||c||: mediaan=%.3f, max=%.3f | gedempte replicaties: %d (%.1f%%)\n",
            median(s2_ratio_log, na.rm = TRUE),
            max(s2_ratio_log, na.rm = TRUE),
            sum(s2_damped), 100*mean(s2_damped)))

# Afstand tot optimum
ok_dist <- conv_default & is.finite(dist_ij1)
cat(sprintf("\n─── AFSTAND STARTWAARDEN TOT OPTIMUM (mediaan) ───\n"))
cat(sprintf("  IJ1:            %.4f\n", median(dist_ij1[ok_dist], na.rm = TRUE)))
cat(sprintf("  HOIJ-2:         %.4f\n", median(dist_ij2[ok_dist], na.rm = TRUE)))
cat(sprintf("  HOIJ-2 + ridge: %.4f\n", median(dist_ij2r[ok_dist], na.rm = TRUE)))
cat(sprintf("  Theta_hat:      %.4f\n",
            median(sqrt(rowSums((theta_default[conv_default, , drop = FALSE] -
                                   matrix(theta_hat, sum(conv_default), n_params,
                                          byrow = TRUE))^2)), na.rm = TRUE)))

# ═══════════════════════════════════════════════════════════════════
# [W12] BREAK-EVEN: setupkosten vs bespaarde GN-iteraties
# Voor GN is een optimizeriteratie niet zonder meer gelijk te stellen aan één
# gradientevaluatie. Daarom worden setup-evaluaties en bespaarde iteraties
# afzonderlijk gerapporteerd. De cruciale vergelijking blijft HOIJ-2 versus
# IJ1, want IJ1 heeft de tweedorde-setup NIET nodig.
# ═══════════════════════════════════════════════════════════════════
n_conv <- sum(same_opt)
saved_ij2_vs_ij1 <- (med_ij1 - med_ij2) * n_conv
saved_ij2_vs_def <- (med_def - med_ij2) * n_conv
cat(sprintf("\n─── [W12] SETUPKOSTEN EN ITERATIEWINST (n=%d replicaties) ───\n", n_conv))
cat(sprintf("  Setup 2e orde:  %d gradientevaluaties + %d casewise-loglik-evaluaties (%.1f s)\n",
            n_grad_setup, n_ll_setup, t_chk + t_H + t_T))
cat(sprintf("  Bespaard t.o.v. Default: ~%.0f GN-iteraties\n",
            saved_ij2_vs_def))
cat(sprintf("  Bespaard t.o.v. IJ1:     ~%.0f GN-iteraties\n",
            saved_ij2_vs_ij1))
cat("  NB: IJ1 vereist geen setup (alleen Scores en H.inv, gratis uit lavaan).\n")
cat("      De 2e-orde-machinerie is voor warm starts alleen te rechtvaardigen\n")
cat("      als de besparing t.o.v. IJ1 de setupkosten overstijgt.\n")

# Plot
par(mfrow = c(1, 2))

# Links: iteraties per methode (boxplot, doorsnede zelfde optimum)     [W6][W13]
iter_df <- data.frame(
  iteraties = c(F_default[same_opt], F_ij1[same_opt], F_ij2[same_opt],
                F_ij2r[same_opt], F_theta_hat[same_opt]),
  methode = factor(rep(c("Default", "IJ1", "HOIJ-2", "HOIJ-2+Ridge", "Theta_hat"),
                       each = sum(same_opt)),
                   levels = c("Default", "IJ1", "HOIJ-2", "HOIJ-2+Ridge", "Theta_hat"))
)
boxplot(iteraties ~ methode, data = iter_df,
        col = c("gray80", "lightskyblue", "steelblue", "royalblue", "salmon"),
        main = sprintf("Iteraties per startwaarde-methode\n(Bifactor, N=%d, zelfde optimum)", SAMPLE_N),  # [W17]
        ylab = "Iteraties tot convergentie", las = 2, cex.axis = 0.8)

# Rechts: afstand tot optimum over tijd                                [W6]
plot(dist_ij1[ok_dist], type = "l", col = "gray50", lwd = 1,
     ylim = range(c(dist_ij1[ok_dist], dist_ij2[ok_dist],
                    dist_ij2r[ok_dist]), na.rm = TRUE),
     ylab = "||θ_start - θ_optimum||", xlab = "Bootstrap replicatie",
     main = "Startwaarde-kwaliteit over tijd")
lines(dist_ij2[ok_dist], col = "red", lwd = 1.5)
lines(dist_ij2r[ok_dist], col = "blue", lwd = 1.5)
abline(v = RIDGE_START, lty = 2, col = "gray")
legend("topright", legend = c("IJ1", "HOIJ-2", "HOIJ-2 + ridge"),
       col = c("gray50", "red", "blue"), lwd = 2, cex = 0.8)
text(RIDGE_START, max(dist_ij2[ok_dist], na.rm = TRUE) * 0.9,
     "ridge start", pos = 4, cex = 0.7, col = "gray40")

par(mfrow = c(1, 1))
