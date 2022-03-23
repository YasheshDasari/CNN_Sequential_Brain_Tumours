import dataset_management
import machine_learning


def main():
    start_program = dataset_management.LoadDataset()
    start_program.manage_directories()
    start_program.store_augmented_data()
    print("Directory creation completed")

    modeling = machine_learning.Modeling()
    modeling.model_definition()
    modeling.predict()


if __name__ == '__main__':
    main()
