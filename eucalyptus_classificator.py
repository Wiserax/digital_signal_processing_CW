from classificator_GLCM import convert_images_to_textural_characteristics, get_accuracy

if __name__ == "__main__":
    class_names = ['1', '2']
    root_folder = './eucalyptus/'
    stat_count = 50

    SKO = 0.1
    iteration_count = 100
    averaged_elements = [10]

    input_train_params = root_folder + 'input_train_params.csv'
    input_test_params = root_folder + 'input_test_params.csv'

    # uncomment line 18 and comment line 18 if you prefer to use params from best_genereated_params.csv
    # input_train_params = root_folder + 'best_generated_params.csv'

    convert_images_to_textural_characteristics(root_folder + 'train/', input_train_params)
    convert_images_to_textural_characteristics(root_folder + 'test/', input_test_params)

    get_accuracy(class_names, root_folder, stat_count,
                 input_train_params, input_test_params,
                 SKO, iteration_count, averaged_elements, is_generate=True)