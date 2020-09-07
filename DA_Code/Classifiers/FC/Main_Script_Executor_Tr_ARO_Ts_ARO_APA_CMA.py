import os

Schedule = []

Schedule.append("python Main_Train_FC114.py --method_type Unet --epochs 100 --batch_size 32 --lr 0.0001 " 
                "--beta1 0.9 --data_augmentation True --vertical_blocks 10 --horizontal_blocks 10 "
                "--fixed_tiles True --defined_before False --image_channels 7 --patches_dimension 128 "
                "--overlap 0.98 --compute_ndvi False --balanced_tr True "
                "--buffer True --buffer_dimension_out 2 --buffer_dimension_in 0 --porcent_of_last_reference_in_actual_reference 100 --porcent_of_positive_pixels_in_actual_reference 2 "
                "--num_classes 2 --phase train --training_type classification --runs 10 " 
                "--patience 10 --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam " 
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image --data_t2_name 21_07_2017_image "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Test_FC114.py --method_type Unet --batch_size 500 --vertical_blocks 10 "
                "--horizontal_blocks 10 --overlap 0.75 --image_channels 7 --patches_dimension 128 --compute_ndvi False --num_classes 2 "
                "--phase test --training_type classification --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_AMAZON_16_17_R232_67_1988_2016_I128_MA_NDVI_FALSE_Adam "
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image --data_t2_name 21_07_2017_image "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Compute_Metrics_05.py --method_type Unet --vertical_blocks 10 "
                "--horizontal_blocks 10 --patches_dimension 128 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text True --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_AMAZON_16_17_R232_67_1988_2016_I128_MA_NDVI_FALSE_Adam "
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image --data_t2_name 21_07_2017_image "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Compute_Average_Metrics_MT.py --method_type DANN --vertical_blocks 10 "
                "--horizontal_blocks 10 --patches_dimension 128 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 --Npoints 100 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text False --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_AMAZON_16_17_R232_67_1988_2016_I128_MA_NDVI_FALSE_Adam "
                "--dataset Amazon_RO --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 18_07_2016_image --data_t2_name 21_07_2017_image "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG32620_R232_67 --reference_t2_name REFERENCE_2017_EPSG32620 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

# Tr amazon RO, Ts Amazon PA
Schedule.append("python Main_Test_FC114.py --method_type Unet --batch_size 500 --vertical_blocks 5 "
                "--horizontal_blocks 3 --overlap 0.75 --image_channels 7 --patches_dimension 128 --compute_ndvi False --num_classes 2 "
                "--phase test --training_type classification --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_AMAZON_16_17_R225_62_1988_2016_I128_MA_NDVI_FALSE_Adam "
                "--dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 02_08_2016_image_R225_62 --data_t2_name 20_07_2017_image_R225_62 "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG4674_R225_62 --reference_t2_name REFERENCE_2017_EPSG4674_R225_62 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Compute_Metrics_05.py --method_type Unet --vertical_blocks 5 "
                "--horizontal_blocks 3 --patches_dimension 128 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text True --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_AMAZON_16_17_R225_62_1988_2016_I128_MA_NDVI_FALSE_Adam "
                "--dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 02_08_2016_image_R225_62 --data_t2_name 20_07_2017_image_R225_62 "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG4674_R225_62 --reference_t2_name REFERENCE_2017_EPSG4674_R225_62 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Compute_Average_Metrics_MT.py --method_type DANN --vertical_blocks 5 "
                "--horizontal_blocks 3 --patches_dimension 128 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 69 --Npoints 100 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text False --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_AMAZON_16_17_R225_62_1988_2016_I128_MA_NDVI_FALSE_Adam "
                "--dataset Amazon_PA --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2016 --data_t2_year 2017 "
                "--data_t1_name 02_08_2016_image_R225_62 --data_t2_name 20_07_2017_image_R225_62 "
                "--reference_t1_name PAST_REFERENCE_FROM_1988_2007_EPSG4674_R225_62 --reference_t2_name REFERENCE_2017_EPSG4674_R225_62 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

# Tr amazon RO, Ts Cerrado MA
Schedule.append("python Main_Test_FC114.py --method_type Unet --batch_size 500 --vertical_blocks 3 "
                "--horizontal_blocks 5 --overlap 0.75 --image_channels 7 --patches_dimension 128 --compute_ndvi False --num_classes 2 "
                "--phase test --training_type classification --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_CERRADO_17_18_R220_63_I128_MA_NDVI_FALSE_Adam "
                "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2017 --data_t2_year 2018 "
                "--data_t1_name 18_08_2017_image --data_t2_name 21_08_2018_image "
                "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674 --reference_t2_name REFERENCE_2018_EPSG4674 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Compute_Metrics_05.py --method_type Unet --vertical_blocks 3 "
                "--horizontal_blocks 5 --patches_dimension 128 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 11 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text True --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_CERRADO_17_18_R220_63_I128_MA_NDVI_FALSE_Adam "
                "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2017 --data_t2_year 2018 "
                "--data_t1_name 18_08_2017_image --data_t2_name 21_08_2018_image "
                "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674 --reference_t2_name REFERENCE_2018_EPSG4674 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

Schedule.append("python Main_Compute_Average_Metrics_MT.py --method_type Unet --vertical_blocks 3 "
                "--horizontal_blocks 5 --patches_dimension 128 --fixed_tiles True --overlap 0.75 --buffer True "
                "--buffer_dimension_out 2 --buffer_dimension_in 0 --eliminate_regions True --area_avoided 11 --Npoints 100 "
                "--compute_ndvi False --phase compute_metrics --training_type classification "
                "--save_result_text False --checkpoint_dir checkpoint_10_run_ft_tr_AMAZON_16_17_R232_67_Unet_1988_2016_I128_NDVI_FALSE_Adam --results_dir results_10_run_ft_tr_AMAZON_16_17_R232_67_1988_2016_ts_CERRADO_17_18_R220_63_I128_MA_NDVI_FALSE_Adam "
                "--dataset Cerrado_MA --images_section Organized/Images/ --reference_section Organized/References/ "
                "--data_type .npy --data_t1_year 2017 --data_t2_year 2018 "
                "--data_t1_name 18_08_2017_image --data_t2_name 21_08_2018_image "
                "--reference_t1_name PAST_REFERENCE_FOR_2018_EPSG4674 --reference_t2_name REFERENCE_2018_EPSG4674 "
                "--dataset_main_path /mnt/DATA/PEDROSOTO/Trabajo_Domain_Adaptation/Dataset/")

for i in range(len(Schedule)):
    os.system(Schedule[i])
