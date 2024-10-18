import matplotlib.pyplot as plt

if __name__ == '__main__':

    # TOKEN CLASSIFICATION
    token_raw_classification = [0.7845984598459846, 0.7848978251143837, 0.7885919727459467, 0.7834107000850724]
    token_nc_classification = [0.7826182618261827, 0.7857132748960767, 0.7901522630329961, 0.7829140973772174]
    token_norm_classification = [0.6792079207920793, 0.6784344771586168, 0.6763328961946393, 0.6722891554858177]

    # TOKEN CLONE DETECTION
    token_raw_50_50 = [0.877, 0.878, 0.876, 0.877]
    token_nc_50_50 = [0.87, 0.869, 0.87, 0.87]
    token_norm_50_50 = [0.737, 0.737, 0.737, 0.737]

    token_raw_25_75 = [0.797, 0.651, 0.817, 0.672]
    token_nc_25_75 = [0.793, 0.677, 0.742, 0.7]  # for these two, I used a screenshot to
    token_norm_25_75 = [0.783, 0.663, 0.656, 0.659]  # capture results so I only copied to 3dp

    token_raw_75_25 = [0.8766666666666667, 0.8221714865550482, 0.8549689440993788, 0.8362807711028186]
    token_nc_75_25 = [0.9133333333333333, 0.8424258474576272, 0.8893405600722675, 0.8628595541177297]
    token_norm_75_25 = [0.7833333333333333, 0.62, 0.7239053516143299, 0.6356434163568079]

    # AST CLASSIFICATION
    ast_nc_classification = [0.9454345434543454, 0.9462693499641897, 0.9469572780750515, 0.9456045941794006]
    ast_norm_classification = [0.9385847797062751, 0.9373931603021092, 0.9390832833023847, 0.9369025183199663]

    # AST CLONE DETECTION

    ast_nc_50_50 = [0.9233333333333333, 0.9232752489331437, 0.9233693477390956, 0.9233120311197556]
    ast_norm_50_50 = [0.8184931506849316, 0.8155793527781692, 0.8272452756672511, 0.8161449361449361]

    ast_nc_25_75 = [0.91, 0.839169836652495, 0.9377301277886073, 0.8735935329827245]
    ast_norm_25_75 = [0.8249158249158249, 0.684765482971761, 0.8097947571631783, 0.7139152341434499]

    ast_nc_75_25 = [0.95, 0.9135586635586636, 0.9380958913505728, 0.9250886451032945]
    ast_norm_75_25 = [0.8419243986254296, 0.7128725929833817, 0.8086632243258749, 0.7421802773497689]

    # GRAPH CLASSIFICATION

    graph_nc_classification = [0.8645925925925926, 0.8653797261337353, 0.8720703137572821, 0.8645748305189122]
    graph_norm_classification = [0.8195884031846177, 0.8140641545972351, 0.8271095567428693, 0.8159549657375561]

    # GRAPH CLONE DETECTION
    graph_nc_50_50 = [0.9209183673469388, 0.9219739001328436, 0.9232859531772575, 0.9208931419457735]
    graph_norm_50_50 = [0.8455284552845529, 0.8489388264669163, 0.8465391545849796, 0.8453649176205915]

    graph_nc_25_75 = [0.9183673469387755, 0.8650056306306306, 0.9080251770259637, 0.8837097192865354]
    graph_norm_25_75 = [0.8475609756097561, 0.7544134907897574, 0.7838928685983106, 0.7672555710438178]

    graph_nc_75_25 = [0.9183673469387755, 0.8827397260273973, 0.8989718691989148, 0.8904071291280797]
    graph_norm_75_25 = [0.8455284552845529, 0.7680107526881721, 0.7964908802537669, 0.7803571428571429]

    # Metrics and their corresponding names
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    raw_recall_token = [token_raw_50_50[2], token_raw_25_75[2], token_raw_75_25[2]]
    nc_recall_token = [token_nc_50_50[2], token_nc_25_75[2], token_nc_75_25[2]]
    norm_recall_token = [token_norm_50_50[2], token_norm_25_75[2], token_norm_75_25[2]]
    recall_full_token = [nc_recall_token, norm_recall_token]

    nc_recall_ast = [ast_nc_50_50[2], ast_nc_25_75[2], ast_nc_75_25[2]]
    norm_recall_ast = [ast_norm_50_50[2], ast_norm_25_75[2], ast_norm_75_25[2]]
    recall_full_ast = [nc_recall_ast, norm_recall_ast]

    nc_recall_graph = [graph_nc_50_50[2], graph_nc_25_75[2], graph_nc_75_25[2]]
    norm_recall_graph = [graph_norm_50_50[2], graph_norm_25_75[2], graph_norm_75_25[2]]
    recall_full_graph = [nc_recall_graph, norm_recall_graph]

    raw = [token_raw_classification[0]]
    non_normalised = [token_nc_classification[0], ast_nc_classification[0], graph_nc_classification[0]]
    normalised = [token_norm_classification[0], ast_norm_classification[0], graph_norm_classification[0]]

    token_graph = raw_recall_token + nc_recall_token + norm_recall_token
    ast_grpah = nc_recall_ast + norm_recall_ast
    graph_graph = nc_recall_graph + norm_recall_graph

    graphs2 = [token_graph, ast_grpah, graph_graph]
    l1 = ['Raw Code', 'Non Normalised', 'Normalised']
    l2 = ['Non Normalised', 'Normalised']
    l3 = ['Non Normalised', 'Normalised']
    splits2 = ['50/50', '25/75', '75/25', ' 50/50', ' 25/75', ' 75/25', '50/50 ', '25/75 ', '75/25 ']
    splits3 = ['50/50', '25/75', '75/25', ' 50/50', ' 25/75', ' 75/25']
    graphs2_lables = [splits2, splits3, splits3]

    first_graph = recall_full_token[0] + recall_full_ast[0] + recall_full_graph[0]
    print(len(first_graph))
    second_graph = recall_full_token[1] + recall_full_ast[1] + recall_full_graph[1]
    graphs = [first_graph, second_graph]
    splits = ['50/50', '25/75', '75/25', ' 50/50', ' 25/75', ' 75/25', '50/50 ', '25/75 ', '75/25 ']
    methods = ['Token Based', 'AST Based', 'CFG Based']
    class_acc_full = [non_normalised, normalised]
    plot_names = ['Token Based', 'AST Based', 'CFG Based']

    # all_scores = [scores_raw, scores_nc, scores_norm]
    conditions = ['Raw Code', 'No Comments', 'Normalised']  # Names for each plot
    colors = ['green', 'blue', 'red']  # Green for raw, blue for nc, red for norm
    colors2 = ['orange', 'yellow', 'red']
    colors3 = ['tomato', 'limegreen', 'royalblue']
    colors4 = ['tomato', 'tomato', 'tomato', 'limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue',
               'royalblue']
    c1 = ['darkgreen', 'darkgreen', 'darkgreen', 'green', 'green', 'green', 'limegreen', 'limegreen', 'limegreen']
    c2 = ['green', 'green', 'green', 'limegreen', 'limegreen', 'limegreen']
    colors5 = [c1, c2, c2]

    # Create subplots: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))  # 1 row, 3 columns of plots

    # Loop through each condition and corresponding scores to plot them
    for i, ax in enumerate(axes):
        ax.bar(graphs2_lables[i], graphs2[i], color=colors5[i])
        # print(all_scores[i])
        ax.set_ylim(0, 1)  # Ensure all bars are on the same scale (0 to 1)
        ax.set_title(plot_names[i], fontsize=16)
        ax.set_ylabel('Recall', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
