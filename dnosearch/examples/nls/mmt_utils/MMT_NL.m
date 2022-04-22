function OUT = MMT_NL(IN,options)
IN = ifft(IN);
OUT = fft(-1i*options.lambda*(abs(IN).^2).*IN +options.F);