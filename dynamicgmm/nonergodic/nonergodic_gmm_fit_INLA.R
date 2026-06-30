#!/usr/bin/env Rscript

suppressPackageStartupMessages({
#libraries
library(sp)
library(hdf5r)
library(configr)
library(optparse)
library(parallel)
#library(rgdal)
library(fields)
library(viridisLite)
library(stringr)
library(assertthat)
library(pracma)
# Bayesian regression
library(INLA)
library(inlabru)
library(posterior)
#plotting packages
library(ggplot2)
library(maps)
})

read_distance_matrix_hdf5 <- function(filepath){
  fle <- H5File$new(filepath, mode="r")
  eqid <- fle[["eqid"]]$read()
  rsn <- fle[["rsn"]]$read()
  ssn <- fle[["ssn"]]$read()
  distances <- t(fle[["distance"]]$read())
  fle$close_all()
  num_cols = ncol(distances)
  distances <- as.data.frame(distances)
  colnames(distances) <- sprintf("c.%g", 1:num_cols)
  distance_matrix <- data.frame(eqid=eqid, ssn=ssn, rsn=rsn)
  distance_matrix <- cbind(distance_matrix, distances)
  return(distance_matrix)
}

#Unique elements
UniqueIdxInv <- function(data_array){
  #' Unique elements, indices and inverse of data_array
  #' 
  #' Input:
  #'  data_array: input array
  #'  
  #' Output:
  #'  unq: unique data
  #'  idx: indices of unique data
  #'  inv: inverse indices for creating original array
  
  #number of data
  n_data <-length(data_array)
  
  #create data data-frame
  df_data <- data.frame(data=data_array)
  #get data-frame with unique data
  df_data_unq <- unique(df_data)
  data_unq    <- df_data_unq$data
  
  #get indices of unique data values
  data_unq_idx <- strtoi(row.names(df_data_unq))
  
  #get inverse indices
  data_unq_inv  <- array(0,n_data)
  for (k in 1:length(data_unq)){
    #return k for element equal to data_unq[k] else 0
    data_unq_inv <- data_unq_inv + ifelse(data_array %in% data_unq[k],k,0)
  }
  
  #return output
  return(list(unq=data_unq, idx=data_unq_idx, inv=data_unq_inv))
}


inla_fit_type1 <- function(config_file, n_threads, verbose){

  # Load in data
  df_flatfile <- read.csv(config$files$input)
  # Set all the other filenames(for export)
  coefficients_file <- config$files$coefficients
  hyper_parameters_file <- config$files$hyperparameters
  hyper_posteriors_file <- config$files$hyperposteriors
  residuals_file <- config$files$residuals
  # Load in general configuration for INLA mesh
  
  # Fix these (maybe?)
  flag_gp_approx <- config$flag_gp_approx # TRUE
  res_name <- config$res_name # "tot"
  # Mesh
  mesh_edge_max <- config$mesh$edge_max # 15
  mesh_inner_offset <- config$mesh$inner_offset # 15
  mesh_outer_offset <- config$mesh$outer_offset # 50

  n_data <- nrow(df_flatfile)
  # setup earthquake data
  data_eq_all <- df_flatfile[,c('eqid','mag','eqX', 'eqY')]
  out_unq  <- UniqueIdxInv(df_flatfile[,'eqid'])
  eq_idx   <- out_unq$idx
  eq_inv   <- out_unq$inv
  data_eq  <- data_eq_all[eq_idx,]
  X_eq     <- data_eq[,c(3,4)] #earthquake coordinates
  X_eq_all <- data_eq_all[,c(3,4)]
  # create earthquake ids for all records (1 to n_eq)
  eq_id <- eq_inv
  n_eq  <- nrow(data_eq)
  
  # setup station data
  data_sta_all <- df_flatfile[,c('ssn','Vs30','staX','staY')]
  out_unq   <- UniqueIdxInv(df_flatfile[,'ssn'])
  sta_idx   <- out_unq$idx
  sta_inv   <- out_unq$inv
  data_sta  <- data_sta_all[sta_idx,]
  X_sta     <- data_sta[,c(3,4)] #station coordinates
  X_sta_all <- data_sta_all[,c(3,4)]
  #create station indices for all records (1 to n_sta)
  sta_id <- sta_inv
  n_sta  <- nrow(data_sta)
  
  #ground-motion observations  
  y_data <- df_flatfile[,res_name]
  
  utm_zone <- unique(df_flatfile$UTMzone)
  utm_no   <- as.numeric(gsub("([0-9]+).*$", "\\1", utm_zone))
  
  #prior on the fixed effects
  prior_fixed <- list(
    mean.intercept = config$fixed_effects$mean_intercept,
    prec.intercept = config$fixed_effects$prec_intercept,
    mean = (list(intcp=config$fixed_effects$mean$itcp,
                 default=config$fixed_effects$mean$default)),
    prec = (list(intcp=config$fixed_effects$prec$itcp,
                 default=config$fixed_effects$mean$default))
    )

  #covariates
  df_inla_covar <- data.frame(intcp = 1, eq = eq_id, sta = sta_id)
  # Generate mesh
  mesh <- fm_mesh_2d_inla(loc=rbind(as.matrix(X_eq),as.matrix(X_sta)) ,
                          max.edge = c(1,5)*mesh_edge_max,
                          cutoff = 3, offset = c(mesh_inner_offset, mesh_outer_offset))
  # Prior distribution on omega for the ds2s independent kernel
  prior_omega_1bs <- list(prec = list(
        prior = config$prior$bs$distribution,
        param = c(config$prior$bs$shape, config$prior$bs$inverse_scale)
        )
    )
  #spde earthquake prior (de)
  spde_eq <- inla.spde2.pcmatern(
      mesh = mesh,
      alpha = config$prior$de$alpha, # Mesh and smoothness parameter
      prior.range = c(config$prior$de$range_0, config$prior$de$p_range_0), # P(range < 100) = 0.95
      prior.sigma = c(config$prior$de$sigma_0, config$prior$de$p_sigma_0) # P(sigma > 0.30) = 0.10
      )  
  
  #spde station prior(as) (for the ds2s spatial kernel)
  spde_sta <- inla.spde2.pcmatern(
    mesh = mesh,
    alpha = config$prior$as$alpha, # Mesh and smoothness parameter
    prior.range = c(config$prior$as$range_0, config$prior$as$p_range_0), # P(range < 100) = 0.95
    prior.sigma = c(config$prior$as$sigma_0, config$prior$as$p_sigma_0) # P(sigma > 0.30) = 0.10
  )  
  
  A_eq    <- inla.spde.make.A(mesh, loc = as.matrix(X_eq_all))
  idx.eq  <- inla.spde.make.index("idx.eq",spde_eq$n.spde)
  A_sta   <- inla.spde.make.A(mesh, loc = as.matrix(X_sta_all))
  idx.sta <- inla.spde.make.index("idx.sta",spde_sta$n.spde)
  
  #prior distributions on phi_0 and tau_0
  prior_phi_0 <- list(prec = list(
    prior = config$prior$phi_0$distribution,
    param = c(config$prior$phi_0$shape, config$prior$phi_0$inverse_scale))
  )
  
  prior_tau_0 <- list(prec = list(
    prior = config$prior$tau_0$distribution,
    param = c(config$prior$tau_0$shape, config$prior$tau_0$inverse_scale))
  )
  
  #functional form (with spatial var)
  form_inla_spatial <- y ~ 0 + intcp + 
    f(eq, model="iid", hyper=prior_tau_0) + f(sta, model="iid", hyper=prior_omega_1bs) +
    f(idx.eq, model = spde_eq) + f(idx.sta, model = spde_sta) 
  
  #build stack
  stk_inla_spatial <- inla.stack(
    data = list(y = y_data),
    A = list(A_eq, A_sta, 1),
    effects = list(idx.eq = idx.eq, idx.sta = idx.sta, df_inla_covar),
    tag = 'model_inla_spatial'
  )

  fit_inla_spatial <- inla(form_inla_spatial, 
                           data = inla.stack.data(stk_inla_spatial),
                           family="gaussian",
                           control.family = list(hyper = list(prec = prior_phi_0)),
                           control.fixed = prior_fixed,
                           control.predictor = list(A = inla.stack.A(stk_inla_spatial)),
                           control.compute = list(dic = TRUE, cpo = TRUE, waic = TRUE),
                           control.inla = list(int.strategy='eb', strategy="gaussian"),
                           verbose=verbose,  num.threads=n_threads)
  # Post-Processing Results
  hyp_param <- data.frame(matrix(ncol = 6, nrow = 0))
  colnames(hyp_param) <- colnames(fit_inla_spatial$summary.hyperpar)
  
  hyp_param['dc_0',]    <- fit_inla_spatial$summary.fixed['intcp',]
  #correlation lengths of spatial terms
  hyp_param['ell_1e',]  <- fit_inla_spatial$summary.hyperpar['Range for idx.eq',]
  hyp_param['ell_1as',] <- fit_inla_spatial$summary.hyperpar['Range for idx.sta',]
  #standard deviations of spatial terms
  hyp_param['omega_1e',]  <- fit_inla_spatial$summary.hyperpar['Stdev for idx.eq',]
  hyp_param['omega_1as',] <- fit_inla_spatial$summary.hyperpar['Stdev for idx.sta',]  
  hyp_param['omega_1bs',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for sta',] ) 
  #aleatory terms
  hyp_param['phi_0',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for the Gaussian observations',] )
  hyp_param['tau_0',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for eq',] )
  #unavailable sd for transformed variables
  hyp_param[c('omega_1bs','phi_0','tau_0'),'sd'] <- NA
  
  prjct_grid_eq  <- inla.mesh.projector(mesh, loc = as.matrix(X_eq))
  prjct_grid_sta <- inla.mesh.projector(mesh, loc = as.matrix(X_sta))
  
  #coefficients    
  coeff_1e  <- fit_inla_spatial$summary.random$idx.eq
  coeff_1as <- fit_inla_spatial$summary.random$idx.sta
  coeff_1bs <- fit_inla_spatial$summary.random$sta
  #coeff mean and std
  coeff_1e_mu   <- inla.mesh.project(prjct_grid_eq,  coeff_1e$mean)
  coeff_1e_sig  <- inla.mesh.project(prjct_grid_eq,  coeff_1e$sd)
  coeff_1as_mu  <- inla.mesh.project(prjct_grid_sta, coeff_1as$mean)
  coeff_1as_sig <- inla.mesh.project(prjct_grid_sta, coeff_1as$sd)
  coeff_1bs_mu  <- coeff_1bs$mean
  coeff_1bs_sig <- coeff_1bs$sd
  
  #mean prediction
  y_new_mu <- hyp_param['dc_0','mean'] + coeff_1e_mu[eq_inv] + coeff_1as_mu[sta_inv] + coeff_1bs_mu[sta_inv] 
  
  #residuals
  res_tot_mu <- y_data - y_new_mu
  res_dB_mu  <- fit_inla_spatial$summary.random$eq$mean[eq_inv]
  res_dWS_mu <- res_tot_mu - res_dB_mu
  
  ## Summarize coefficients and residuals
  # ---------------------------
  df_flatinfo  <- df_flatfile[,c('rsn','eqid','ssn','eqLat','eqLon','staLat','staLon','eqX','eqY','staX','staY')]
  
  #summary coefficients
  df_coeff <- data.frame(rsn=df_flatinfo$rsn,
                         dc_0_mean=hyp_param['dc_0','mean'],
                         dc_1e_mean=coeff_1e_mu[eq_inv],  
                         dc_1as_mean=coeff_1as_mu[sta_inv],
                         dc_1bs_mean=coeff_1bs_mu[sta_inv], 
                         dc_0_sig=hyp_param['dc_0','sd'], 
                         dc_1e_sig=coeff_1e_sig[eq_inv], 
                         dc_1as_sig=coeff_1as_sig[sta_inv], 
                         dc_1bs_sig=coeff_1bs_sig[sta_inv])
  df_coeff <- merge(df_flatinfo, df_coeff, by=c('rsn'))
  
  #summary predictions and residuals
  df_predict_summary <- data.frame(rsn=df_flatinfo$rsn, nerg_mu=y_new_mu, 
                                   res_tot=res_tot_mu, res_between=res_dB_mu, res_within=res_dWS_mu)
  df_predict_summary <- merge(df_flatinfo, df_predict_summary, by=c('rsn'))
  
  # Calculate the hyper-posterior distribution
  # ---------------------------
  #intercept
  post_dc_0 <- as.data.frame(fit_inla_spatial$marginals.fixed$intcp)
  #aleatory parameters
  post_phi_0 <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for the Gaussian observations']]))
  post_tau_0 <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for eq']]))
  #non-ergodic scales
  post_omega_1e  <- as.data.frame(inla.tmarginal(function(x) exp( x),   fit_inla_spatial$internal.marginals.hyperpar[['log(Stdev) for idx.eq']]))
  post_omega_1as <- as.data.frame(inla.tmarginal(function(x) exp( x),   fit_inla_spatial$internal.marginals.hyperpar[['log(Stdev) for idx.sta']]))
  post_omega_1bs <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for sta']]))
  #correlation length
  post_ell_1e   <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla_spatial$internal.marginals.hyperpar[['log(Range) for idx.eq']]))
  post_ell_1as  <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla_spatial$internal.marginals.hyperpar[['log(Range) for idx.sta']]))
  
  #compute posterior cdfs
  post_dc_0$y_int      <- cumtrapz(post_dc_0$x, post_dc_0$y)   / trapz(post_dc_0$x, post_dc_0$y)
  post_phi_0$y_int     <- cumtrapz(post_phi_0$x, post_phi_0$y) / trapz(post_phi_0$x, post_phi_0$y)
  post_tau_0$y_int     <- cumtrapz(post_tau_0$x, post_tau_0$y) / trapz(post_tau_0$x, post_tau_0$y)
  post_omega_1e$y_int  <- cumtrapz(post_omega_1e$x, post_omega_1e$y)   / trapz(post_omega_1e$x, post_omega_1e$y)
  post_omega_1as$y_int <- cumtrapz(post_omega_1as$x, post_omega_1as$y) / trapz(post_omega_1as$x, post_omega_1as$y)
  post_omega_1bs$y_int <- cumtrapz(post_omega_1bs$x, post_omega_1bs$y) / trapz(post_omega_1bs$x, post_omega_1bs$y)
  post_ell_1e$y_int    <- cumtrapz(post_ell_1e$x, post_ell_1e$y)       / trapz(post_ell_1e$x, post_ell_1e$y)
  post_ell_1as$y_int   <- cumtrapz(post_ell_1as$x, post_ell_1as$y)     / trapz(post_ell_1as$x, post_ell_1as$y)
  
  #posterior distributions
  #define quantiles
  hyp_posterior <- data.frame(quant=seq(0.0,1.0,0.01))
  #compute pdf and cdf
  if (! all(is.na(post_dc_0$y_int))){
    hyp_posterior$dc_0          <- approx(post_dc_0$y_int,      post_dc_0$x,      hyp_posterior$quant)$y
    hyp_posterior$dc_0_pdf      <- approx(post_dc_0$y_int,      post_dc_0$y,      hyp_posterior$quant)$y
  } else {
    hyp_posterior$dc_0          <- NaN
    hyp_posterior$dc_0_pdf      <- NaN
  }
  if (! all(is.na(post_ell_1e$y_int))){
    hyp_posterior$ell_1e        <- approx(post_ell_1e$y_int,    post_ell_1e$x,    hyp_posterior$quant)$y
    hyp_posterior$ell_1e_pdf    <- approx(post_ell_1e$y_int,    post_ell_1e$y,    hyp_posterior$quant)$y
  } else {
    hyp_posterior$ell_1e        <- NaN
    hyp_posterior$ell_1e_pdf    <- NaN
  }
  if (! all(is.na(post_ell_1as$y_int))){  
    hyp_posterior$ell_1as       <- approx(post_ell_1as$y_int,   post_ell_1as$x,   hyp_posterior$quant)$y
    hyp_posterior$ell_1as_pdf   <- approx(post_ell_1as$y_int,   post_ell_1as$y,   hyp_posterior$quant)$y
  } else {
    hyp_posterior$ell_1as       <- NaN
    hyp_posterior$ell_1as_pdf   <- NaN
  }
  if (! all(is.na(post_omega_1e$y_int))){  
    hyp_posterior$omega_1e      <- approx(post_omega_1e$y_int,  post_omega_1e$x,  hyp_posterior$quant)$y
    hyp_posterior$omega_1e_pdf  <- approx(post_omega_1e$y_int,  post_omega_1e$y,  hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1e      <- NaN
    hyp_posterior$omega_1e_pdf  <- NaN
  }
  if (! all(is.na(post_omega_1as$y_int))){  
    hyp_posterior$omega_1as     <- approx(post_omega_1as$y_int, post_omega_1as$x, hyp_posterior$quant)$y
    hyp_posterior$omega_1as_pdf <- approx(post_omega_1as$y_int, post_omega_1as$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1as     <- NaN
    hyp_posterior$omega_1as_pdf <- NaN
  }
  if (! all(is.na(post_omega_1bs$y_int))){  
    hyp_posterior$omega_1bs     <- approx(post_omega_1bs$y_int, post_omega_1bs$x, hyp_posterior$quant)$y
    hyp_posterior$omega_1bs_pdf <- approx(post_omega_1bs$y_int, post_omega_1bs$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1bs     <- NaN
    hyp_posterior$omega_1bs_pdf <- NaN
  }
  if  (! all(is.na(post_phi_0$y_int))){ 
    hyp_posterior$phi_0         <- approx(post_phi_0$y_int,     post_phi_0$x,     hyp_posterior$quant)$y
    hyp_posterior$phi_0_pdf     <- approx(post_phi_0$y_int,     post_phi_0$y,     hyp_posterior$quant)$y
  } else {
    hyp_posterior$phi_0         <- NaN
    hyp_posterior$phi_0_pdf     <- NaN
  }
  if  (! all(is.na(post_tau_0$y_int))){  
    hyp_posterior$tau_0         <- approx(post_tau_0$y_int,     post_tau_0$x,     hyp_posterior$quant)$y
    hyp_posterior$tau_0_pdf     <- approx(post_tau_0$y_int,     post_tau_0$y,     hyp_posterior$quant)$y
  } else {
    hyp_posterior$tau_0         <- NaN
    hyp_posterior$tau_0_pdf     <- NaN
  }
  
  # Export
  write.csv(df_coeff, coefficients_file)
  write.csv(hyp_param, hyper_parameters_file)
  write.csv(hyp_posterior, hyper_posteriors_file)
  write.csv(df_predict_summary, residuals_file)
}

inla_fit_type2 <- function(config, n_threads, verbose){
  # Load in data
  df_flatfile <- read.csv(config$files$input)
  df_cellinfo <- read.csv(config$files$cell_info)
  df_cellmat <- read_distance_matrix_hdf5(config$files$cell_mat)
  # Set all the other filenames(for export)
  coefficients_file <- config$files$coefficients
  hyper_parameters_file <- config$files$hyperparameters
  hyper_posteriors_file <- config$files$hyperposteriors
  residuals_file <- config$files$residuals
  attenuation_file <- config$files$attenuation
  
  flag_gp_approx <- config$flag_gp_approx # TRUE
  res_name <- config$res_name # "tot"
  c_a_arg <- config$c_a_ergodic # Ergodic attenuation coefficient
  # Load in general configuration for INLA mesh

  # Mesh
  mesh_edge_max <- config$mesh$edge_max # 15
  mesh_inner_offset <- config$mesh$inner_offset # 15
  mesh_outer_offset <- config$mesh$outer_offset # 50
  n_data <- nrow(df_flatfile)
  # setup earthquake data
  data_eq_all <- df_flatfile[,c('eqid','mag','eqX', 'eqY')]
  out_unq  <- UniqueIdxInv(df_flatfile[,'eqid'])
  eq_idx   <- out_unq$idx
  eq_inv   <- out_unq$inv
  data_eq  <- data_eq_all[eq_idx,]
  X_eq     <- data_eq[,c(3,4)] #earthquake coordinates
  X_eq_all <- data_eq_all[,c(3,4)]
  # create earthquake ids for all records (1 to n_eq)
  eq_id <- eq_inv
  n_eq  <- nrow(data_eq)
  
  # setup station data
  data_sta_all <- df_flatfile[,c('ssn','Vs30','staX','staY')]
  out_unq   <- UniqueIdxInv(df_flatfile[,'ssn'])
  sta_idx   <- out_unq$idx
  sta_inv   <- out_unq$inv
  data_sta  <- data_sta_all[sta_idx,]
  X_sta     <- data_sta[,c(3,4)] #station coordinates
  X_sta_all <- data_sta_all[,c(3,4)]
  #create station indices for all records (1 to n_sta)
  sta_id <- sta_inv
  n_sta  <- nrow(data_sta)
  
  #ground-motion observations  
  y_data <- df_flatfile[,res_name]
  
  #cell data
  #keep only cell distance for records in df_flatfile
  df_cellmat <- df_cellmat[match(df_flatfile$rsn, df_cellmat$rsn), ]
  assert_that(nrow(df_flatfile) == nrow(df_cellmat))
  #cell info
  cell_names_all <- colnames(df_cellmat)
  cell_names_all <- cell_names_all[str_detect(cell_names_all,'c.')]
  cell_ids_all   <- as.integer( str_extract(cell_names_all,'\\d+') )
  #cells with crossing paths
  cell_valid     <- colSums(df_cellmat[,cell_names_all])  > 0
  # cell_valid[]   <- TRUE
  cell_names     <- cell_names_all[cell_valid]
  cell_ids       <- cell_ids_all[cell_valid]
  
  #distance matrix
  RC <- as.matrix(df_cellmat[,cell_names])
  RC_sparse <- as(RC ,"dgCMatrix") #sparse matrix
  # print( paste('max R_rup misfit', max(abs(rowSums(RC) - df_flatfile$Rrup))) )
  
  #UTM zone
  utm_zone <- unique(df_flatfile$UTMzone)
  utm_no   <- as.numeric(gsub("([0-9]+).*$", "\\1", utm_zone))
  
  
  #prior on the fixed effects
  prior_fixed <- list(
    mean.intercept = config$fixed_effects$mean_intercept,
    prec.intercept = config$fixed_effects$prec_intercept,
    mean = (list(intcp=config$fixed_effects$mean$itcp,
                 R=config$fixed_effects$mean$R,
                 default=config$fixed_effects$mean$default)),
    prec = (list(intcp=config$fixed_effects$prec$itcp,
                 R=config$fixed_effects$prec$R,
                 default=config$fixed_effects$mean$default))
  )
  
  #covariates
  df_inla_covar <- data.frame(intcp=1, R=df_flatfile$Rrup, 
                              eq=eq_id, sta=sta_id)

  # Generate mesh
  mesh <- fm_mesh_2d_inla(loc=rbind(as.matrix(X_eq),as.matrix(X_sta)) ,
                          max.edge = c(1,5)*mesh_edge_max,
                          cutoff = 3, offset = c(mesh_inner_offset, mesh_outer_offset))
  # Prior distribution on omega for the ds2s independent kernel
  prior_omega_1bs <- list(prec = list(
    prior = config$prior$bs$distribution,
    param = c(config$prior$bs$shape, config$prior$bs$inverse_scale)
  )
  )
  #spde earthquake prior (de)
  spde_eq <- inla.spde2.pcmatern(
    mesh = mesh,
    alpha = config$prior$de$alpha, # Mesh and smoothness parameter
    prior.range = c(config$prior$de$range_0, config$prior$de$p_range_0), # P(range < 100) = 0.95
    prior.sigma = c(config$prior$de$sigma_0, config$prior$de$p_sigma_0) # P(sigma > 0.30) = 0.10
  )  
  
  #spde station prior(as) (for the ds2s spatial kernel)
  spde_sta <- inla.spde2.pcmatern(
    mesh = mesh,
    alpha = config$prior$as$alpha, # Mesh and smoothness parameter
    prior.range = c(config$prior$as$range_0, config$prior$as$p_range_0), # P(range < 100) = 0.95
    prior.sigma = c(config$prior$as$sigma_0, config$prior$as$p_sigma_0) # P(sigma > 0.30) = 0.10
  )  
  
  A_eq    <- inla.spde.make.A(mesh, loc = as.matrix(X_eq_all))
  idx.eq  <- inla.spde.make.index("idx.eq",spde_eq$n.spde)
  A_sta   <- inla.spde.make.A(mesh, loc = as.matrix(X_sta_all))
  idx.sta <- inla.spde.make.index("idx.sta",spde_sta$n.spde)

  #cell-specific anelastic attenuation
  #---   ---   ---   ---   ---   ---
  prior_omega_ca <- list(prec = list(
    prior = 'pc.prec',
    param = c(config$prior$cell$u, config$prior$cell$alpha))
  ) 
  
  #cell ids
  df_inla_covar$idx_cell <- 1:nrow(df_inla_covar)
  
  #prior distributions on phi_0 and tau_0
  prior_phi_0 <- list(prec = list(
    prior = config$prior$phi_0$distribution,
    param = c(config$prior$phi_0$shape, config$prior$phi_0$inverse_scale))
  )
  
  prior_tau_0 <- list(prec = list(
    prior = config$prior$tau_0$distribution,
    param = c(config$prior$tau_0$shape, config$prior$tau_0$inverse_scale))
  )
  
  #functional form (with spatial var)
  form_inla_spatial <- y ~ 0 + intcp + R +
    f(eq, model="iid", hyper=prior_tau_0) + f(sta, model="iid", hyper=prior_omega_1bs) +
    f(idx.eq, model = spde_eq) + f(idx.sta, model = spde_sta) + 
    f(idx_cell, model = "z", Z = RC_sparse, hyper=prior_omega_ca)
  
  #build stack
  stk_inla_spatial <- inla.stack(data = list(y = y_data),
                                 A = list(A_eq, A_sta, 1),
                                 effects = list(idx.eq = idx.eq,
                                                idx.sta = idx.sta,
                                                df_inla_covar),
                                 tag = 'model_inla_spatial')
  #fit inla model
  fit_inla_spatial <- inla(form_inla_spatial,
                           data = inla.stack.data(stk_inla_spatial),
                           family="gaussian",
                           control.family = list(hyper = list(prec = prior_phi_0)),
                           control.fixed = prior_fixed,
                           control.predictor = list(A = inla.stack.A(stk_inla_spatial)),
                           control.compute = list(dic = TRUE, cpo = TRUE, waic = TRUE),
                           control.inla = list(int.strategy='eb', strategy="gaussian"),
                           verbose=verbose, num.threads=n_threads)
  ## Post-processing Results
  # ---------------------------
  #hyper-parameters
  hyp_param <- data.frame(matrix(ncol = 6, nrow = 0))
  colnames(hyp_param) <- colnames(fit_inla_spatial$summary.hyperpar)
  
  hyp_param['dc_0',]    <- fit_inla_spatial$summary.fixed['intcp',]
  #correlation lengths of spatial terms
  hyp_param['ell_1e',]  <- fit_inla_spatial$summary.hyperpar['Range for idx.eq',]
  hyp_param['ell_1as',] <- fit_inla_spatial$summary.hyperpar['Range for idx.sta',]
  #standard deviations of spatial terms
  hyp_param['omega_1e',]  <- fit_inla_spatial$summary.hyperpar['Stdev for idx.eq',]
  hyp_param['omega_1as',] <- fit_inla_spatial$summary.hyperpar['Stdev for idx.sta',]  
  hyp_param['omega_1bs',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for sta',] ) 
  #anelastic attenuation
  hyp_param['mu_cap',]    <- fit_inla_spatial$summary.fixed['R',]
  hyp_param['omega_cap',] <- 1/sqrt(fit_inla_spatial$summary.hyperpar['Precision for idx_cell',] ) 
  #aleatory terms
  hyp_param['phi_0',] <- 1/sqrt( fit_inla_spatial$summary.hyperpar['Precision for the Gaussian observations',] )
  hyp_param['tau_0',] <- 1/sqrt( fit_inla_spatial$summary.hyperpar['Precision for eq',] )
  #unavailable sd for transformed variables
  hyp_param[c('omega_1bs','omega_cap','phi_0','tau_0'),'sd'] <- NA
  
  #projections
  prjct_grid_eq  <- inla.mesh.projector(mesh, loc = as.matrix(X_eq))
  prjct_grid_sta <- inla.mesh.projector(mesh, loc = as.matrix(X_sta))
  
  #coefficients    
  coeff_1e  <- fit_inla_spatial$summary.random$idx.eq
  coeff_1as <- fit_inla_spatial$summary.random$idx.sta
  coeff_1bs <- fit_inla_spatial$summary.random$sta
  #coeff mean and std
  coeff_1e_mu   <- inla.mesh.project(prjct_grid_eq,  coeff_1e$mean)
  coeff_1e_sig  <- inla.mesh.project(prjct_grid_eq,  coeff_1e$sd)
  coeff_1as_mu  <- inla.mesh.project(prjct_grid_sta, coeff_1as$mean)
  coeff_1as_sig <- inla.mesh.project(prjct_grid_sta, coeff_1as$sd)
  coeff_1bs_mu  <- coeff_1bs$mean
  coeff_1bs_sig <- coeff_1bs$sd
  #cell specific anelastic attenuation
  cell_atten <- fit_inla_spatial$summary.random$idx_cell[-(1:n_data),]
  #cell mean and std
  cap_mu  <- cell_atten$mean      + hyp_param['mu_cap','mean']
  cap_sig <- sqrt(cell_atten$sd^2 + hyp_param['mu_cap','sd']^2)
  #effect of anelastic attenuation in GM
  cells_Lcap_mu  <- RC %*% cap_mu
  cells_Lcap_sig <- sqrt(RC^2 %*% cap_sig^2)
  
  #mean prediction
  y_new_mu <- hyp_param['dc_0','mean'] + coeff_1e_mu[eq_inv] + coeff_1as_mu[sta_inv] + coeff_1bs_mu[sta_inv] + cells_Lcap_mu
  
  #residuals
  res_tot_mu <- y_data - y_new_mu
  res_dB_mu  <- fit_inla_spatial$summary.random$eq$mean[eq_inv]
  res_dWS_mu <- res_tot_mu - res_dB_mu
  
  ## Summarize coefficients and residuals
  # ---------------------------
  df_flatinfo  <- df_flatfile[,c('rsn','eqid','ssn','eqLat','eqLon','staLat','staLon','eqX','eqY','staX','staY')]
  
  #summary coefficients
  df_coeff <- data.frame(rsn=df_flatinfo$rsn,
                         dc_0_mean=hyp_param['dc_0','mean'],
                         dc_1e_mean=coeff_1e_mu[eq_inv],  
                         dc_1as_mean=coeff_1as_mu[sta_inv],
                         dc_1bs_mean=coeff_1bs_mu[sta_inv], 
                         dc_0_sig=hyp_param['dc_0','sd'], 
                         dc_1e_sig=coeff_1e_sig[eq_inv], 
                         dc_1as_sig=coeff_1as_sig[sta_inv], 
                         dc_1bs_sig=coeff_1bs_sig[sta_inv])
  df_coeff <- merge(df_flatinfo, df_coeff, by=c('rsn'))
  
  #summary predictions and residuals
  df_predict_summary <- data.frame(rsn=df_flatinfo$rsn, nerg_mu=y_new_mu, 
                                   res_tot=res_tot_mu, res_between=res_dB_mu, res_within=res_dWS_mu)
  df_predict_summary <- merge(df_flatinfo, df_predict_summary, by=c('rsn'))
  
  #summary attenuation cells
  df_catten_summary <- data.frame(cellid=cell_ids, c_cap_mean=cap_mu, c_cap_sig=cap_sig)
  df_catten_summary <- merge(df_cellinfo[c('cellid','cellname','mptLat','mptLon','mptX','mptY','mptZ','UTMzone')],
                             df_catten_summary, by=c('cellid'))

  ## Posterior distributions
  # ---------------------------
  #intercept
  post_dc_0 <- as.data.frame(fit_inla_spatial$marginals.fixed$intcp)
  #aleatory parameters
  post_phi_0 <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for the Gaussian observations']]))
  post_tau_0 <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for eq']]))
  #non-ergodic scales
  post_omega_1e  <- as.data.frame(inla.tmarginal(function(x) exp( x),   fit_inla_spatial$internal.marginals.hyperpar[['log(Stdev) for idx.eq']]))
  post_omega_1as <- as.data.frame(inla.tmarginal(function(x) exp( x),   fit_inla_spatial$internal.marginals.hyperpar[['log(Stdev) for idx.sta']]))
  post_omega_1bs <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for sta']]))
  #correlation length
  post_ell_1e   <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla_spatial$internal.marginals.hyperpar[['log(Range) for idx.eq']]))
  post_ell_1as  <- as.data.frame(inla.tmarginal(function(x) exp( x), fit_inla_spatial$internal.marginals.hyperpar[['log(Range) for idx.sta']]))
  #cell specific attenuation
  post_omega_cap <- as.data.frame(inla.tmarginal(function(x) exp(-x/2), fit_inla_spatial$internal.marginals.hyperpar[['Log precision for idx_cell']]))
  
  #compute posterior cdfs
  post_dc_0$y_int      <- cumtrapz(post_dc_0$x, post_dc_0$y)   / trapz(post_dc_0$x, post_dc_0$y)
  post_phi_0$y_int     <- cumtrapz(post_phi_0$x, post_phi_0$y) / trapz(post_phi_0$x, post_phi_0$y)
  post_tau_0$y_int     <- cumtrapz(post_tau_0$x, post_tau_0$y) / trapz(post_tau_0$x, post_tau_0$y)
  post_omega_1e$y_int  <- cumtrapz(post_omega_1e$x, post_omega_1e$y)   / trapz(post_omega_1e$x, post_omega_1e$y)
  post_omega_1as$y_int <- cumtrapz(post_omega_1as$x, post_omega_1as$y) / trapz(post_omega_1as$x, post_omega_1as$y)
  post_omega_1bs$y_int <- cumtrapz(post_omega_1bs$x, post_omega_1bs$y) / trapz(post_omega_1bs$x, post_omega_1bs$y)
  post_ell_1e$y_int    <- cumtrapz(post_ell_1e$x, post_ell_1e$y)       / trapz(post_ell_1e$x, post_ell_1e$y)
  post_ell_1as$y_int   <- cumtrapz(post_ell_1as$x, post_ell_1as$y)     / trapz(post_ell_1as$x, post_ell_1as$y)
  post_omega_cap$y_int <- cumtrapz(post_omega_cap$x, post_omega_cap$y) / trapz(post_omega_cap$x, post_omega_cap$y)
  
  #posterior distributions
  #define quantiles
  hyp_posterior <- data.frame(quant=seq(0.0,1.0,0.01))
  #compute pdf and cdf
  if (! all(is.na(post_dc_0$y_int))){
    hyp_posterior$dc_0          <- approx(post_dc_0$y_int,      post_dc_0$x,      hyp_posterior$quant)$y
    hyp_posterior$dc_0_pdf      <- approx(post_dc_0$y_int,      post_dc_0$y,      hyp_posterior$quant)$y
  } else {
    hyp_posterior$dc_0          <- NaN
    hyp_posterior$dc_0_pdf      <- NaN
  }
  if (! all(is.na(post_ell_1e$y_int))){
    hyp_posterior$ell_1e        <- approx(post_ell_1e$y_int,    post_ell_1e$x,    hyp_posterior$quant)$y
    hyp_posterior$ell_1e_pdf    <- approx(post_ell_1e$y_int,    post_ell_1e$y,    hyp_posterior$quant)$y
  } else {
    hyp_posterior$ell_1e        <- NaN
    hyp_posterior$ell_1e_pdf    <- NaN
  }
  if (! all(is.na(post_ell_1as$y_int))){  
    hyp_posterior$ell_1as       <- approx(post_ell_1as$y_int,   post_ell_1as$x,   hyp_posterior$quant)$y
    hyp_posterior$ell_1as_pdf   <- approx(post_ell_1as$y_int,   post_ell_1as$y,   hyp_posterior$quant)$y
  } else {
    hyp_posterior$ell_1as       <- NaN
    hyp_posterior$ell_1as_pdf   <- NaN
  }
  if (! all(is.na(post_omega_1e$y_int))){  
    hyp_posterior$omega_1e      <- approx(post_omega_1e$y_int,  post_omega_1e$x,  hyp_posterior$quant)$y
    hyp_posterior$omega_1e_pdf  <- approx(post_omega_1e$y_int,  post_omega_1e$y,  hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1e      <- NaN
    hyp_posterior$omega_1e_pdf  <- NaN
  }
  if (! all(is.na(post_omega_1as$y_int))){  
    hyp_posterior$omega_1as     <- approx(post_omega_1as$y_int, post_omega_1as$x, hyp_posterior$quant)$y
    hyp_posterior$omega_1as_pdf <- approx(post_omega_1as$y_int, post_omega_1as$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1as     <- NaN
    hyp_posterior$omega_1as_pdf <- NaN
  }
  if (! all(is.na(post_omega_1bs$y_int))){  
    hyp_posterior$omega_1bs     <- approx(post_omega_1bs$y_int, post_omega_1bs$x, hyp_posterior$quant)$y
    hyp_posterior$omega_1bs_pdf <- approx(post_omega_1bs$y_int, post_omega_1bs$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_1bs     <- NaN
    hyp_posterior$omega_1bs_pdf <- NaN
  }
  if  (! all(is.na(post_phi_0$y_int))){  
    hyp_posterior$phi_0         <- approx(post_phi_0$y_int,     post_phi_0$x,     hyp_posterior$quant)$y
    hyp_posterior$phi_0_pdf     <- approx(post_phi_0$y_int,     post_phi_0$y,     hyp_posterior$quant)$y
  } else {
    hyp_posterior$phi_0         <- NaN
    hyp_posterior$phi_0_pdf     <- NaN
  }
  if  (! all(is.na(post_tau_0$y_int))){  
    hyp_posterior$tau_0         <- approx(post_tau_0$y_int,     post_tau_0$x,     hyp_posterior$quant)$y
    hyp_posterior$tau_0_pdf     <- approx(post_tau_0$y_int,     post_tau_0$y,     hyp_posterior$quant)$y
  } else {
    hyp_posterior$tau_0         <- NaN
    hyp_posterior$tau_0_pdf     <- NaN
  }
  if  (! all(is.na(post_omega_cap$y_int))){  
    hyp_posterior$omega_cap     <- approx(post_omega_cap$y_int, post_omega_cap$x, hyp_posterior$quant)$y
    hyp_posterior$omega_cap_pdf <- approx(post_omega_cap$y_int, post_omega_cap$y, hyp_posterior$quant)$y
  } else {
    hyp_posterior$omega_cap     <- NaN
    hyp_posterior$omega_cap_pdf <- NaN
  }
  
  write.csv(df_coeff, coefficients_file)
  write.csv(as.data.frame(hyp_param), hyper_parameters_file)
  write.csv(hyp_posterior, hyper_posteriors_file)
  write.csv(df_predict_summary, residuals_file)
  write.csv(df_catten_summary, file=attenuation_file)
}

parser <- OptionParser()
parser <- add_option(parser, c("-c", "--config"), action="store", type="character")
parser <- add_option(parser, c("-n", "--num_cores"), action="store", type="integer", default=detectCores())
parser <- add_option(parser, c("-v", "--verbose"), action="store_true", type="logical", default=FALSE)

opt <- parse_args(parser)
# Load config
config <- read.config(opt$config)
# Choose the model type (1, 2, ... etc)
print(opt$num_cores)
print(opt$verbose)
switch (
  as.character(config$type),
  "1" = inla_fit_type1(config, opt$num_cores, opt$verbose),
  "2" = inla_fit_type2(config, opt$num_cores, opt$verbose),
)

