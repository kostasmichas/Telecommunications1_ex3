
% 1. Generate a binary sequence
N = 100;
bit_seq = randi([0 1], 1, 4*N);

% Plotting the binary sequence
stem(bit_seq);
xlabel('Position');
ylabel('Bit');

% 2. Convert the binary sequence to 16-PSK symbols

[X, X_i_n, X_q_n] = bits_to_PSK16(bit_seq)

% Plotting the real and imaginary parts of the symbols
figure;
stem(X_i_n, 'r'); % Real part
grid on;
hold on;
title('Real');
stem(X_q_n, 'b'); % Imaginary part
title('Imaginary');
hold off;
grid off;


%3
T = 10^(-2);
over = 10;
Ts = T / over;
A = 4;
a = 0.5;
Fs = 1 / Ts;
Nf = 2048;

% Generating SRRC pulse
[phi, t] = srrc_pulse(T, over, A, a);

% Upsample the signal X and apply pulse shaping
Xi_n_d = 1 / Ts * upsample(X_i_n, over);
Xq_n_d = 1 / Ts * upsample(X_q_n, over);

% Time vectors for plotting
t_X = 0:Ts:N*T-Ts;
t_conv = min(t) + min(t_X):Ts:max(t) + max(t_X);

% Convolve with the SRRC pulse
Xi_n_t = conv(phi, Xi_n_d);
Xq_n_t = conv(phi, Xq_n_d);

% Plotting the time-domain signals
figure();
plot(t_conv, Xi_n_t);
grid on;
title('X_I_N');

figure();
plot(t_conv, Xq_n_t);
grid on;
title('X_Q_N');

% Total duration of the signal
T_total = max(t_conv) - min(t_conv);

% Frequency vector for the periodogram
F = -Fs/2:(Fs/Nf):(Fs/2)-(Fs/Nf);

% Calculate the periodogram using FFT
FX_IN = fftshift(fft(Xi_n_t, Nf) * Ts);
FX_QN = fftshift(fft(Xq_n_t, Nf) * Ts);

% Calculate the power spectral density
PX_IN = (abs(FX_IN).^2) / T_total;
PX_QN = (abs(FX_QN).^2) / T_total;

% Plotting the periodogram
figure();
plot(F, PX_IN);
grid on;
title('PERIODOGRAM OF Xi_n');

figure();
plot(F, PX_QN);
grid on;
title('PERIODOGRAM OF Xq_n');

% 4. Modulate the symbols with carriers and plot the modulated signals

% Carrier frequency
F0 = 200;

% Generate cosine and sine carriers
cos_carrier = 2*cos(2*pi*F0*t_conv)';
sine_carrier = -2*sin(2*pi*F0*t_conv)';

% Modulate the symbols with carriers
XIt = Xi_n_t .* cos_carrier;
XQt = Xq_n_t .* sine_carrier;

% Plotting the modulated signals
figure();
plot(t_conv, XIt);
grid on;
title('Modulated Signal (I-component)');
xlabel('Time');
ylabel('Amplitude');

figure();
plot(t_conv, XQt);
grid on;
title('Modulated Signal (Q-component)');
xlabel('Time');
ylabel('Amplitude');

% Calculate the periodograms of the modulated signals
FX_I = fftshift(fft(XIt, Nf) * Ts);
FX_Q = fftshift(fft(XQt, Nf) * Ts);

PX_I = (abs(FX_I).^2) / T_total;
PX_Q = (abs(FX_Q).^2) / T_total;

% Plotting the periodograms
figure();
plot(F, PX_I);
grid on;
title('Periodogram of Modulated Signal (I-component)');
xlabel('Frequency');
ylabel('Power Spectral Density');

figure();
plot(F, PX_Q);
grid on;
title('Periodogram of Modulated Signal (Q-component)');
xlabel('Frequency');
ylabel('Power Spectral Density');



% 5. Combine the modulated signals and calculate the periodogram

% Combine the modulated signals to calculate channel input
Xt = XIt + XQt;

% Plotting the combined signal
figure();
plot(t_conv, Xt);
grid on;
title('Combined Modulated Signal');
xlabel('Time');
ylabel('Amplitude');

% Calculate the periodogram of the combined signal
X_F = FX_IN + FX_QN;

P_X = abs(X_F).^2 / T_total;

% Plotting the periodogram
figure();
plot(F, P_X);
grid on;
title('Periodogram of Combined Signal');
xlabel('Frequency');
ylabel('Power Spectral Density');



% 7. Add noise to the combined signal

% Signal-to-Noise Ratio (SNR) in decibels
SNRdB = 10;

% Calculate the variance of the white Gaussian noise
v_w = 1 / (Ts * 10^(SNRdB/10)); % Convert SNR to variance

% Calculate the standard deviation of the noise
v_n = Ts * v_w / 2; % Convert variance to standard deviation

% Generate white Gaussian noise samples
W_t = sqrt(v_n) * randn(length(t_conv), 1); % Generate noise samples

% Add the noise to the combined signal
Y_t = Xt + W_t; % Add noise to the combined signal

%8

Y_i_t = Y_t.*cos_carrier;
Y_q_t = Y_t.*sine_carrier;

t_conv_new = min(t_conv) + min(t) : Ts : max(t_conv) + max(t); % Define new time vector
T_total_new = max(t_conv_new) - min(t_conv_new); % Calculate total duration

%9

Y_i_filtered = conv(Y_i_t, phi) * Ts; % Convolve Y_i_t with pulse shaping filter
Y_q_filtered = conv(Y_q_t, phi) * Ts; % Convolve Y_q_t with pulse shaping filter


figure();
plot(t_conv_new, Y_i_filtered, 'r'); % Plot filtered Y_i_t
grid on;
title('FILTERED Y1_t');

figure();
plot(t_conv_new, Y_q_filtered, 'k'); % Plot filtered Y_q_t
grid on;
title('FILTERED Y2_t');

Y_i_filtered_F = fftshift(fft(Y_i_filtered, Nf) * Ts); % Perform Fourier transform on filtered Y_i_t
Y_q_filtered_F = fftshift(fft(Y_q_filtered, Nf) * Ts); % Perform Fourier transform on filtered Y_q_t


P_Y_i_filtered = (abs(Y_i_filtered_F).^2) / T_total_new; % Calculate power spectral density of Y_i_t
P_Y_q_filtered = (abs(Y_q_filtered_F).^2) / T_total_new; % Calculate power spectral density of Y_q_t

figure();
plot(F, P_Y_i_filtered); % Plot periodogram of Y_i_t
grid on;
title('PERIODOGRAM OF Y1_t');

figure();
plot(F, P_Y_q_filtered); % Plot periodogram of Y_q_t
grid on;
title('PERIODOGRAM OF Y2_t');

% 10. Sample the output of the adapted filters at the appropriate time instances and plot the output sequence Y using scatterplot

% Determine the sampling instances based on the symbol rate
Ts_symbol = Ts; % Symbol duration
t_symbol = 0:Ts_symbol:T_total_new-Ts_symbol; % Time instances for sampling

% Sample the filtered signals at the symbol rate
Y_i_sampled = Y_i_filtered(1:over:end); % Sampled Y_i_filtered
Y_q_sampled = Y_q_filtered(1:over:end); % Sampled Y_Q_filtered

% Create a scatterplot of the sampled output sequence Y
scatterplot(complex(Y_i_sampled, Y_q_sampled));
title('Scatterplot of the Output Sequence Y');

%b1
j=1;
for SNRdB=-2:2:16
    sum_num_symbol_errors=0;
    sum_num_bit_errors=0;
    for K=1:1000
        N=100;
        bit_seq = randi([0 1], 1, 4*N);
        X=bits_to_PSK16(bit_seq);
        T=10^-2;
        over=10;
        Ts=T/over;
        A=4;
        a=0.5;
        Fs=1/Ts;
        Nf=2048;
        DT=Fs/Nf;
        [phi, t] = srrc_pulse(T, over, A, a);

        % Upsample the signal X and apply pulse shaping
        Xi_n_d = 1 / Ts * upsample(X_i_n, over);
        Xq_n_d = 1 / Ts * upsample(X_q_n, over);
        
        % Time vectors for plotting
        t_X = 0:Ts:N*T-Ts;
        t_conv = min(t) + min(t_X):Ts:max(t) + max(t_X);
        
        % Convolve with the SRRC pulse
        Xi_n_t = conv(phi, Xi_n_d);
        Xq_n_t = conv(phi, Xq_n_d);
        
                F0 = 200;
        
        % Generate cosine and sine carriers
        cos_carrier = 2*cos(2*pi*F0*t_conv)';
        sine_carrier = -2*sin(2*pi*F0*t_conv)';
        
        % Modulate the symbols with carriers
        XIt = Xi_n_t .* cos_carrier;
        XQt = Xq_n_t .* sine_carrier;

        
        Xt = XIt + XQt;

       % Calculate the variance of the white Gaussian noise
        v_w = 1 / (Ts * 10^(SNRdB/10)); % Convert SNR to variance
        
        % Calculate the standard deviation of the noise
        v_n = Ts * v_w / 2; % Convert variance to standard deviation
        
        % Generate white Gaussian noise samples
        W_t = sqrt(v_n) * randn(length(t_conv), 1); % Generate noise samples
        
        % Add the noise to the combined signal
        Y_t = Xt + W_t; % Add noise to the combined signal
       
        
        Y_i_t = Y_t.*cos_carrier;
        Y_q_t = Y_t.*sine_carrier;

        Y_i_filtered = conv(Y_i_t, phi) * Ts; % Convolve Y_i_t with pulse shaping filter
        Y_q_filtered = conv(Y_q_t, phi) * Ts; % Convolve Y_q_t with pulse shaping filter

      
        l=1;
        for k=2*A*over+1:over:length(t_conv_new)-2*A*over
           Y(l,1)=Y_i_filtered(k);
           Y(l,2)=Y_q_filtered(k);
           l=l+1;
        end

        [est_X,est_bit_seq]=detect_PSK16(Y);
        num_symbol_errors1 = symbol_errors(est_X,X);
        num_bit_errors1 = bit_errors(est_bit_seq, bit_seq); 
        sum_num_symbol_errors= sum_num_symbol_errors + num_symbol_errors1;
        sum_num_bit_errors= sum_num_bit_errors + num_bit_errors1;
    end
    PEsymbol(j)=sum_num_symbol_errors/(N*K);
    PEbit(j)=sum_num_bit_errors/(3*N*K);
    j=j+1;
end


%b2

i=1;
for SNRdB=-2:2:16     
     Pfsymbol(i)=0.75*erfc(sqrt(0.2*(10^(SNRdB/10))));
     i=i+1;
end 
SNRdB=-2:2:16;
figure();
semilogy(SNRdB,PEsymbol);
hold on;
semilogy(SNRdB,Pfsymbol);
grid on;
legend('PEsymbol','Pfsymbol');


%b3
Pfbit=Pfsymbol./2;
figure();
semilogy(SNRdB,PEbit);
hold on;
semilogy(SNRdB,Pfbit);
grid on;
legend('PEbit','Pfbit');

