#!/usr/bin/env python
import argparse
import logging
import sys
import torch
import os
import json
# my staff
from utils.data_helper import FewShotRawDataLoader
from utils.preprocessor import make_dict, save_feature, load_feature, make_preprocessor, make_label_mask
from utils.opt import define_args, basic_args, train_args, test_args, preprocess_args, model_args, option_check
from utils.device_helper import prepare_model, set_device_environment
from utils.trainer import FewShotTrainer, SchemaFewShotTrainer, prepare_optimizer, prepare_few_shot_optimizer
from utils.tester import FewShotTester, SchemaFewShotTester
from utils.model_helper import make_model, load_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_training_data_and_feature(opt, data_loader, preprocessor):
    """ prepare feature and data """
    if opt.load_feature:
        try:
            train_features, train_label2id, train_id2label, train_trans_mat = load_feature(opt.train_path.replace('.json', '.saved.pk'))
            dev_features, dev_label2id, dev_id2label, dev_trans_mat = load_feature(opt.dev_path.replace('.json', '.saved.pk'))
        except FileNotFoundError:
            # Not a saved feature file yet, make it
            opt.load_feature = False
            opt.save_feature = True
            train_features, train_label2id, train_id2label, train_trans_mat, \
                dev_features, dev_label2id, dev_id2label, dev_trans_mat =\
                get_training_data_and_feature(opt, data_loader, preprocessor)
            # restore option
            opt.load_feature = True
            opt.save_feature = False
    else:
        train_examples, train_max_len, train_max_support_size, train_trans_mat = data_loader.load_data(path=opt.train_path)
        dev_examples, dev_max_len, dev_max_support_size, dev_trans_mat = data_loader.load_data(path=opt.dev_path)

        train_label2id, train_id2label = make_dict(train_examples)
        dev_label2id, dev_id2label = make_dict(dev_examples)
        logger.info(' Finish train dev prepare dict ')

        train_features = preprocessor.construct_feature(
            train_examples, train_max_support_size, train_label2id, train_id2label)
        dev_features = preprocessor.construct_feature(
            dev_examples, dev_max_support_size, dev_label2id, dev_id2label)
        logger.info(' Finish prepare train dev features ')
        if opt.save_feature:
            save_feature(opt.train_path.replace('.json', '.saved.pk'),
                         train_features, train_label2id, train_id2label, train_trans_mat)
            save_feature(opt.dev_path.replace('.json', '.saved.pk'), dev_features, dev_label2id, dev_id2label, dev_trans_mat)
    return train_features, train_label2id, train_id2label, train_trans_mat, \
        dev_features, dev_label2id, dev_id2label, dev_trans_mat


def get_testing_data_feature(opt, data_loader, preprocessor):
    """ prepare feature and data """
    if opt.load_feature:
        try:
            test_features, test_label2id, test_id2label, test_trans_mat = \
                load_feature(opt.test_path.replace('.json', '.saved.pk'))
        except FileNotFoundError:
            # Not a saved feature file yet, make it
            opt.load_feature = False
            opt.save_feature = True
            test_features, test_label2id, test_id2label, test_trans_mat = \
                get_testing_data_feature(opt, data_loader, preprocessor)
            # restore option
            opt.load_feature = True
            opt.save_feature = False
    else:
        test_examples, test_max_len, test_max_support_size, test_trans_mat = data_loader.load_data(path=opt.test_path)
        test_label2id, test_id2label = make_dict(test_examples)
        logger.info(' Finish prepare test dict')
        test_features = preprocessor.construct_feature(
            test_examples, test_max_support_size, test_label2id, test_id2label)
        logger.info(' Finish prepare test feature')
        if opt.save_feature:
            save_feature(opt.test_path.replace('.json', '.saved.pk'),
                         test_features, test_label2id, test_id2label, test_trans_mat)
    return test_features, test_label2id, test_id2label, test_trans_mat


def main():
    """ to start the experiment """
    ''' set option '''
    parser = argparse.ArgumentParser()
    parser = define_args(parser, basic_args, train_args, test_args, preprocess_args, model_args)
    opt = parser.parse_args()
    print('Args:\n', json.dumps(vars(opt), indent=2))
    opt = option_check(opt)

    ''' device & environment '''
    device, n_gpu = set_device_environment(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    logger.info("Environment: device {}, n_gpu {}".format(device, n_gpu))

    ''' data & feature '''
    data_loader = FewShotRawDataLoader(debugging=opt.do_debug)
    preprocessor = make_preprocessor(opt)
    if opt.do_train:
        train_features, train_label2id, train_id2label, train_trans_mat, \
            dev_features, dev_label2id, dev_id2label, dev_trans_mat = \
            get_training_data_and_feature(opt, data_loader, preprocessor)
        # todo: remove the train label mask out of opt.
        if opt.mask_transition:
            opt.train_label_mask = make_label_mask(opt, opt.train_path, train_label2id)
            opt.dev_label_mask = make_label_mask(opt, opt.dev_path, dev_label2id)
            opt.train_trans_mat = [torch.Tensor(item).to(device) for item in train_trans_mat]
            opt.dev_trans_mat = [torch.Tensor(item).to(device) for item in dev_trans_mat]
    else:
        train_features, train_label2id, train_id2label, dev_features, dev_label2id, dev_id2label = [None] * 6
        if opt.mask_transition:
            opt.train_label_mask = None
            opt.dev_label_mask = None
    if opt.do_predict:
        test_features, test_label2id, test_id2label, test_trans_mat = get_testing_data_feature(opt, data_loader, preprocessor)
        if opt.mask_transition:
            opt.test_label_mask = make_label_mask(opt, opt.test_path, test_label2id)
            opt.test_trans_mat = [torch.Tensor(item).to(device) for item in test_trans_mat]
    else:
        test_features, test_label2id, test_id2label = [None] * 3
        if opt.mask_transition:
            opt.test_label_mask = None

    ''' over fitting test '''
    if opt.do_overfit_test:
        test_features, test_label2id, test_id2label = train_features, train_label2id, train_id2label
        dev_features, dev_label2id, dev_id2label = train_features, train_label2id, train_id2label

    ''' select training & testing mode '''
    trainer_class = SchemaFewShotTrainer if opt.use_schema else FewShotTrainer
    tester_class = SchemaFewShotTester if opt.use_schema else FewShotTester

    ''' training '''
    best_model = None
    if opt.do_train:
        logger.info("***** Perform training *****")
        training_model = make_model(opt, num_tags=len(train_label2id), trans_r=1)  # trans_r is 1 for training
        training_model = prepare_model(opt, training_model, device, n_gpu)
        if opt.mask_transition:
            training_model.label_mask = opt.train_label_mask.to(device)
        if opt.upper_lr > 0:  # use different learning rate for upper structure parameter
            param_to_optimize, optimizer = prepare_few_shot_optimizer(opt, training_model, len(train_features))
        else:
            param_to_optimize, optimizer = prepare_optimizer(opt, training_model, len(train_features))
        tester = tester_class(opt, device, n_gpu)
        trainer = trainer_class(opt, optimizer, param_to_optimize, device, n_gpu, tester=tester)
        if opt.warmup_epoch > 0:
            training_model.no_embedder_grad = True

            if opt.upper_lr > 0:  # use different learning rate for upper structure parameter
                stage_1_param_to_optimize, stage_1_optimizer = prepare_few_shot_optimizer(opt, training_model, len(train_features))
            else:
                stage_1_param_to_optimize, stage_1_optimizer = prepare_optimizer(opt, training_model, len(train_features))

            stage_1_trainer = trainer_class(opt, stage_1_optimizer, stage_1_param_to_optimize, device, n_gpu, tester=None)
            trained_model, best_dev_score, test_score = stage_1_trainer.do_train(
                training_model, train_features, opt.warmup_epoch)
            training_model = trained_model
            training_model.no_embedder_grad = False
            print('========== Stage one training finished! ==========')
        trained_model, best_dev_score, test_score = trainer.do_train(
            training_model, train_features, opt.num_train_epochs,
            dev_features, dev_id2label, test_features, test_id2label, best_dev_score_now=0)

        # decide the best model
        if not opt.eval_when_train:  # select best among check points
            best_model, best_score, test_score_then = trainer.select_model_from_check_point(
                train_id2label, dev_features, dev_id2label, test_features, test_id2label, rm_cpt=opt.delete_checkpoint)
        else:  # best model is selected during training
            best_model = trained_model
        logger.info('dev:{}, test:{}'.format(best_dev_score, test_score))
        print('dev:{}, test:{}'.format(best_dev_score, test_score))

    ''' testing '''
    if opt.do_predict:
        logger.info("***** Perform testing *****")
        tester = tester_class(opt, device, n_gpu)
        if not best_model:
            if not opt.saved_model_path:
                raise ValueError("No model trained and no trained model file given!")
            if os.path.isdir(opt.saved_model_path):
                all_cpt_file = list(filter(lambda x: '.cpt.pl' in x, os.listdir(opt.saved_model_path)))
                all_cpt_file = sorted(all_cpt_file,
                                      key=lambda x: int(x.replace('model.step', '').replace('.cpt.pl', '')))
                max_score = 0
                for cpt_file in all_cpt_file:
                    cpt_model = load_model(os.path.join(opt.saved_model_path, cpt_file))
                    testing_model = tester.clone_model(cpt_model, test_id2label)
                    if opt.mask_transition:
                        testing_model.label_mask = opt.test_label_mask.to(device)
                    test_score = tester.do_test(testing_model, test_features, test_id2label, log_mark='test_pred')
                    if test_score > max_score:
                        max_score = test_score
                    logger.info('cpt_file:{} - test:{}'.format(cpt_file, test_score))
                print('max_score:{}'.format(max_score))
            else:
                if not os.path.exists(opt.saved_model_path):
                    logger.info('The model is not exits')
                    raise ValueError('The model is not exits')
                best_model = load_model(opt.saved_model_path)
        if not os.path.isdir(opt.saved_model_path):
            testing_model = tester.clone_model(best_model, test_id2label)  # copy reusable params
            if opt.mask_transition:
                testing_model.label_mask = opt.test_label_mask.to(device)
            test_score = tester.do_test(testing_model, test_features, test_id2label, log_mark='test_pred')
            logger.info('test:{}'.format(test_score))
            print('test:{}'.format(test_score))


if __name__ == "__main__":
    main()
