function [ log_mae, log_rmse ] = calc_log_pdf_errors(pz1, pz2, zz, trunc_level)
%CALC_LOG_PDF_ERRORS Summary of this function goes here
%   Detailed explanation goes here

    if (trunc_level == 0)
        trunc_level = 1e-13;
    end

    pz1t = max(pz1, trunc_level);
    pz2t = max(pz2, trunc_level);

    dz = zz(2) - zz(1);

    use_log10 = true;

    if use_log10
        dp = abs(log10(pz2t) - log10(pz1t));
    else
        dp = abs(log(pz2t) - log(pz1t));
    end

    log_mae = dz*sum(dp);
    log_rmse = dz*sqrt(sum(dp.^2));

end

