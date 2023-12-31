clear

DDPG = 'result_ADMM_GAP_ddpg_2023-11-02_22:04:32_traces.mat';
TD3 = 'result_ADMM_GAP_td3_2023-11-02_22:04:22_traces.mat';

algo = 'TD3';
target = TD3;

load(target)

x = 1:10;
utility_static = repmat(utility_static, 1, 10);

hold on  
plot(x, utility_static, 'b-', 'LineWidth', 2);
plot(x, utility_opt, 'r-', 'LineWidth', 2);
plot(x, utility, 'g-', 'LineWidth', 2);

title(algo);

xlabel('Iterations');
xlim([1, max(x)]);
xticks(1:max(x));

ylabel('Utility');
%ylim([0, 1.5e4]);

legend('Utility Static', 'Utility OPT', 'Utility');

grid on

hold off

filename = [algo, '.png'];

saveas(gcf, filename);
