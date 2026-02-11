from evaluation.evaluator import evaluate
from evaluation.template_level_analysis import evaluate_template_level, evaluate_template_level_lstm
from evaluation.PA_calculator import calculate_parsing_accuracy, calculate_parsing_accuracy_lstm
from varparser.utils import process_template
import pandas as pd
import time
import csv
import os


def evaluator(dataset, model, shot, parse_time, invoc_num, invoc_token, invoc_time):
    parsed_csv = pd.read_csv(f"parsed/{dataset}_{shot}_parsed.csv")
    parsed_csv = parsed_csv.applymap(process_template)

    parsed_result = pd.DataFrame(parsed_csv['parsed_result'])
    ground_truth = pd.DataFrame(parsed_csv['ground_truth'])

    parsed_result.columns = ['EventTemplate']
    ground_truth.columns = ['EventTemplate']
    # print(parsed_result.head())

    print("Start compute grouping accuracy")
    # calculate grouping accuracy
    start_time = time.time()
    GA, FGA = evaluate(ground_truth, parsed_result)

    GA_end_time = time.time() - start_time
    print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(
        GA_end_time))

    # calculate parsing accuracy
    print("Start compute parsing accuracy")
    start_time = time.time()
    PA = calculate_parsing_accuracy(ground_truth, parsed_result)
    PA_end_time = time.time() - start_time
    print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(
        PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(
        dataset, ground_truth, parsed_result)
    TA_end_time = time.time() - start_time
    print(
        'Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

    result_csv = f"result/{model}_{shot}_result.csv"
    if not os.path.exists(result_csv):
        with open(result_csv, 'w', newline='') as csv_file:
            fw = csv.writer(csv_file, delimiter=',')
            fw.writerow(['Dataset', 'parse_time', 'invoc_time', 'invoc_num', 'invoc_token', 'identified_templates',
                        'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])

    result = dataset + ',' + \
        "{:.2f}".format(parse_time) + ',' + \
        "{:.2f}".format(invoc_time) + ',' + \
        "{:.2f}".format(invoc_num) + ',' + \
        "{:.2f}".format(invoc_token) + ',' + \
        str(tool_templates) + ',' + \
        str(ground_templates) + ',' + \
        "{:.4f}".format(GA) + ',' + \
        "{:.4f}".format(PA) + ',' + \
        "{:.4f}".format(FGA) + ',' + \
        "{:.4f}".format(PTA) + ',' + \
        "{:.4f}".format(RTA) + ',' + \
        "{:.4f}".format(FTA) + '\n'

    with open(result_csv, 'a') as summary_file:
        summary_file.write(result)

    return
