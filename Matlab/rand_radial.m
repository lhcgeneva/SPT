function randr = rand_radial(D,dt,randomValues)

    %% draw random variable from distribution r.*exp(r.^2/(4*drand*dt));
    rvec = 0 : 0.0001 : 10;

    % Draw random diffusion constant
    a = 0.01;                  % Variance of diffusion constants
    D = 0.2;               % Mean of diffusion constants                             
    drand = a.*randn(1) + D; % Diffusion rates for each reaction drawn from gaussian distribution with mean d and variance a

    % The probability distribution of the step
    % size is the gaussian in spherical
    % coordinates after integrating out the angle
    pd = @(r) r.*exp(-r.^2/(4*drand*dt));

    % Evaluate the integral from r=0 to r=Inf.
    q = integral(pd,0,Inf);

    % Normalize the function to have an area of 1.0 under it
    pdf = @(r) r.*exp(-r.^2/(4*drand*dt))/q;
    p = integral(pdf,0,Inf);

% Check your distributions
%     figure()
%     fplot(pd,[0.0,2.0])
%     figure()
%     fplot(pdf,[0.0,2.0])
%     figure()
%     plot(rvec,pdf(rvec))

    % the integral of PDF is the cumulative distribution function
    cdf = cumsum(pdf(rvec))/max(cumsum(pdf(rvec))); %returns the cumulative distribution function 
    % of the probability distribution object, pd, evaluated at the values in x.
%     figure()
%     plot(rvec,cdf)

    % remove non-unique elements
    [cdf, mask] = unique(cdf);
    rvec = rvec(mask);

    % inverse interpolation to achieve P(x) -> x projection of the random values
    randr = interp1(cdf, rvec, randomValues);

    % Check your distributions
%    hist(randr, 100);
end