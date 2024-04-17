#adapted from deepSDF code base

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import numpy as np
import DDF
import DDF.workspace as ws
from preprocessData import getSamplesName
import sys
import socket
import random
class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))

class ExpLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor
    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch / self.interval))


# def decayed_learning_rate(step):
# return initial_learning_rate * decay_rate ^ (step / decay_steps)

class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length

def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Exp":
            schedules.append(
                ExpLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def load_latent_vectors(experiment_directory, filename, lat_vecs, ref_col = False):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        if not ref_col == True:
            lat_vecs.load_state_dict(data["latent_codes"])
        else:
            print(data["latent_codes"])
            lat_vecs.weight.data[0, :].copy_(data["latent_codes"]['weight'][0, :])
            lat_vecs.weight.data[-1, :].copy_(data["latent_codes"]['weight'][-1, :])

    return data["epoch"]

def save_logs(
    experiment_directory,
    loss_geom,
    timing_log,
    epoch,
):
    torch.save(
        {
            "epoch": epoch,
            "loss_geom": loss_geom,
            "timing": timing_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )

def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    print(full_filename)
    data = torch.load(full_filename)

    return (
        data["loss_geom"],
        data["timing"],
        data["epoch"]
    )


def clip_logs(loss_geom, loss_text, timing_log, epoch):

    iters_per_epoch = len(loss_log) // len(loss_geom)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    loss_geom = loss_geom[: (iters_per_epoch * epoch)]
    loss_text = loss_text[: (iters_per_epoch * epoch)]
    timing_log = timing_log[:epoch]

    return (loss_geom, loss_text, timing_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())

def setupProcsDist(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    if port == None:
        os.environ['MASTER_PORT'] = '12789'
    else:
        os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    return

def getDataset(experiment_directory):
    specs = ws.load_experiment_specifications(experiment_directory)
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    num_samp_per_scene = specs["SamplesPerScene"]
    generalpreprocessing = get_spec_with_default(specs, "generalpreprocessing", False)
    missingDirs = get_spec_with_default(specs, "missingDirs", False)
    loadDataToRam = get_spec_with_default(specs, "loadDataToRam", False)
    numDataSamplesPerScene = get_spec_with_default(specs, "numDataSamplesPerScene", -1)
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = DDF.data.SDFSamples(
        data_source, train_split, num_samp_per_scene,
        appendName=getSamplesName(experiment_directory,generalpreprocessing),load_ram=loadDataToRam,missingDirs=missingDirs,numDataSamplesPerScene=numDataSamplesPerScene,
    )
    logging.info("Loaded SDF dataset")
    return sdf_dataset


def main_function(rank, distributedtrain, world_size, freePort, sdf_dataset, experiment_directory, continue_from, batch_split):
    if (world_size > 1) and distributedtrain:
        loggerFunc = print
    else:
        loggerFunc = logging.info

    def toCuda(objData):
        if distributedtrain:
            return objData.to(rank)
        else:
            return objData.cuda()

    loggerFunc("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)
    loggerFunc("Experiment description: \n" + str(specs["Description"]))

    if specs["NetworkArchDDF"] == "models":
        arch_ddf = __import__("lfn." + specs["NetworkArchDDF"], fromlist=["LightFieldModel"])
    else:
        arch_ddf = __import__("networks." + specs["NetworkArchDDF"], fromlist=["DDFNet"])
    print(specs["NetworkArchSDF"])
    if not specs["NetworkArchSDF"] is None:
        arch_sdf = __import__("networks." + specs["NetworkArchSDF"], fromlist=["SDFNet"])
        no_net_sdf = False
    else:
        arch_sdf = None
        no_net_sdf = True
    
    loggerFunc(specs["NetworkSpecsDDF"])
    loggerFunc(specs["NetworkSpecsSDF"])

    latent_size = specs["CodeLength"]
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.append(specs["NumEpochs"])
    checkpoints.sort()
    print("Saving checkpoints {}".format(checkpoints))


    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        loggerFunc("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        save_model(experiment_directory, "latest_ddf.pth", decoder_ddf, epoch)
        save_optimizer(experiment_directory, "latest_ddf.pth", optimizer_ddf, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)
        if not no_net_sdf:
            save_model(experiment_directory, "latest_sdf.pth", decoder_sdf, epoch)
            save_optimizer(experiment_directory, "latest_sdf.pth", optimizer_sdf, epoch)
        if lat_vecs_seperate_ddf:
            save_latent_vectors(experiment_directory, "latest_ddf.pth", lat_vecs_ddf, epoch)

    def save_checkpoints(epoch):
        if not no_net_sdf:
            save_model(experiment_directory, str(epoch) + "_sdf.pth", decoder_sdf, epoch)
            save_optimizer(experiment_directory, str(epoch) + "_sdf.pth", optimizer_sdf, epoch)
        save_model(experiment_directory, str(epoch) + "_ddf.pth", decoder_ddf, epoch)
        save_optimizer(experiment_directory, str(epoch) + "_ddf.pth", optimizer_ddf, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)
        if lat_vecs_seperate_ddf:
            save_latent_vectors(experiment_directory, str(epoch) + "_ddf.pth", lat_vecs_ddf, epoch)


    def signal_handler(sig, frame):
        loggerFunc("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch, makezero=False, inSchedule=None):

        for i, param_group in enumerate(optimizer.param_groups):
            if makezero:
                param_group["lr"] = 0
            else:
                sched = inSchedule if inSchedule is not None else i
                param_group["lr"] = lr_schedules[sched].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]

    #default options for codes

    num_epochs = specs["NumEpochs"]
    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    sdfLossWeight = get_spec_with_default(specs, "sdfLossWeight", 1)
    code_bound = get_spec_with_default(specs, "CodeBound", None)
    missingDirs = get_spec_with_default(specs, "missingDirs", False)
    optimizerMode = get_spec_with_default(specs, "optimizerMode", "adam")
    tvlregfeats = get_spec_with_default(specs, "tvlregfeats", False)
    tvlregweight = get_spec_with_default(specs, "tvlregweight", 1.0)
    tvlregweightddf = get_spec_with_default(specs, "tvlregweightddf", tvlregweight)
    sdf_ddf_const_train = get_spec_with_default(specs, "sdf_ddf_const_train", False)
    sdf_ddf_const_wt = get_spec_with_default(specs, "sdf_ddf_const_wt", 0.0)
    ddf_loss_wt = get_spec_with_default(specs, "ddf_loss_wt", 1.0)
    detach_ddf_sdf = get_spec_with_default(specs, "detach_ddf_sdf", False)
    lat_vecs_seperate_ddf = get_spec_with_default(specs, "lat_vecs_seperate_ddf", False)
     
    
     
    
    #we use latent_size/3 for each spaces
    
    if specs["NetworkArchDDF"] == "models":
        decoder_ddf = toCuda(arch_ddf.LightFieldModel(latent_size, **specs["NetworkSpecsDDF"]))
    else:
        decoder_ddf = toCuda(arch_ddf.DDFNet(latent_size, **specs["NetworkSpecsDDF"]))
        
    # decoder_ddf = toCuda(arch_ddf.PointBases(latent_size, **specs["NetworkSpecsPoints"]))
    print("DDF")
    print(decoder_ddf)
    #since no expression space for color, 2/3 *latent size
    if not no_net_sdf:
        decoder_sdf = toCuda(arch_sdf.SDFNet(int(latent_size), **specs["NetworkSpecsSDF"]))
        decoder_sdf_cloned = toCuda(arch_sdf.SDFNet(int(latent_size), **specs["NetworkSpecsSDF"]))
    else:
        decoder_sdf = None
    print("SDF")
    print(decoder_sdf)
    loggerFunc("training with {} GPU(s)".format(torch.cuda.device_count()))


    if (world_size > 1) and distributedtrain:
        setupProcsDist(rank,world_size,freePort)
        decoder_ddf = toCuda(torch.nn.parallel.DistributedDataParallel(decoder_ddf, device_ids=[rank]))
        decoder_sdf = toCuda(torch.nn.parallel.DistributedDataParallel(decoder_sdf, device_ids=[rank]))
        if detach_ddf_sdf:
            decoder_sdf_cloned = toCuda(torch.nn.parallel.DistributedDataParallel(decoder_sdf_cloned, device_ids=[rank]))
    else:
        decoder_ddf = toCuda(torch.nn.DataParallel(decoder_ddf))
        if not no_net_sdf:
            decoder_sdf = toCuda(torch.nn.DataParallel(decoder_sdf))
            if detach_ddf_sdf:
                decoder_sdf_cloned = toCuda(torch.nn.DataParallel(decoder_sdf_cloned))


    num_scenes = len(sdf_dataset)
    print("********************NUMSCENES******************",num_scenes)
    if num_scenes > 1:
        log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    else:
        log_frequency = get_spec_with_default(specs, "LogFrequency", 500)

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    loggerFunc("loading data with {} threads".format(num_data_loader_threads))
    if num_scenes == 1:
        sdf_loader = sdf_dataset
    else:
        if (world_size > 1) and distributedtrain:
            sdf_data_sampler = torch.utils.data.distributed.DistributedSampler(dataset=sdf_dataset, num_replicas=world_size, rank=rank)
            shuffledataset = False
        else:
            sdf_data_sampler = None
            shuffledataset=True
        sdf_loader = data_utils.DataLoader(
            sdf_dataset,
            batch_size=scene_per_batch,
            shuffle=shuffledataset,
            prefetch_factor=16,
            num_workers=num_data_loader_threads,
            drop_last=False, pin_memory=True,
            sampler=sdf_data_sampler
        )
    loggerFunc("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)
    loggerFunc("There are {} scenes".format(num_scenes))

    # create latent vectors of size int(latent_size/3)
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    loggerFunc(
        "initialized geometric latent codes with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    if lat_vecs_seperate_ddf:
        lat_vecs_ddf = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
        torch.nn.init.normal_(
            lat_vecs_ddf.weight.data,
            0.0,
            get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
        )

        loggerFunc(
            "initialized ddf latent codes with mean magnitude {}".format(
                get_mean_latent_vector_magnitude(lat_vecs_ddf)
            )
        )


    loss_l1_None = torch.nn.L1Loss(reduction="none")
    loss_bce = torch.nn.BCELoss()
    
    lrSchedPoints = 0
    loggerFunc("Using this schedule for dsdf: {}, {}".format(lr_schedules[lrSchedPoints].get_learning_rate(0),lr_schedules[lrSchedPoints].interval,lr_schedules[lrSchedPoints].factor))
    if optimizerMode == "adam":
        optParams = [
                {
                    "params": decoder_ddf.parameters(),
                    "lr": lr_schedules[lrSchedPoints].get_learning_rate(0),
                },
            ]
        if lat_vecs_seperate_ddf:
            optParams.append(
                    {
                        "params": lat_vecs_ddf.parameters(),
                        "lr": lr_schedules[1].get_learning_rate(0),
                    })
        optimizer_ddf = torch.optim.Adam(optParams)
        
        
    if not no_net_sdf:
        optimizer_sdf = torch.optim.Adam(
            [
                {
                    "params": decoder_sdf.parameters(),
                    "lr": lr_schedules[0].get_learning_rate(0),
                },
                {
                    "params": lat_vecs.parameters(),
                    "lr": lr_schedules[1].get_learning_rate(0),
                },
            ]
        )

    loss_log = []
    timing_log = []
    start_epoch = 1

    if continue_from is not None:

        loggerFunc('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )
        if lat_vecs_seperate_ddf:
            lat_epoch_2 = load_latent_vectors(
                experiment_directory, continue_from + "_ddf.pth", lat_vecs_ddf
            )
        model_epoch_ddf = ws.load_model_parameters(
            experiment_directory, continue_from, decoder_ddf, False, True, False
        )
        optimizer_epoch_ddf = load_optimizer(
            experiment_directory, continue_from + "_ddf.pth", optimizer_ddf
        )

        if  not no_net_sdf:
            model_epoch_sdf = ws.load_model_parameters(
                experiment_directory, continue_from, decoder_sdf, True, False, False
            )
        else:
            model_epoch_sdf = model_epoch_ddf

        if  not no_net_sdf:
            optimizer_epoch_sdf = load_optimizer(
                experiment_directory, continue_from + "_sdf.pth", optimizer_sdf
            )
        else:
            optimizer_epoch_sdf = optimizer_epoch_ddf

        loss_log, timing_log, log_epoch = load_logs(
            experiment_directory
        )

        start_epoch = model_epoch_sdf + 1
        if not (model_epoch_sdf == optimizer_epoch_sdf and model_epoch_sdf == lat_epoch and model_epoch_ddf==optimizer_epoch_ddf and model_epoch_ddf==lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch_sdf, optimizer_epoch_sdf, lat_epoch, log_epoch
                )
            )
        start_epoch = model_epoch_sdf + 1
        loggerFunc("loaded saved networks")
    else:
        start_epoch = 1

    loggerFunc("starting from epoch {}".format(start_epoch))

    loggerFunc(
        "Number of DDF network parameters: {}".format(
            sum(p.data.nelement() for p in decoder_ddf.parameters())
        )
    )
    if not no_net_sdf:
        loggerFunc(
            "Number of directions network parameters: {}".format(
                sum(p.data.nelement() for p in decoder_sdf.parameters())
            )
        )
    loggerFunc(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    if lat_vecs_seperate_ddf:
        loggerFunc(
            "Number of shape code parameters DDF: {} (# codes {}, code dim {})".format(
                lat_vecs_ddf.num_embeddings * lat_vecs_ddf.embedding_dim,
                lat_vecs_ddf.num_embeddings,
                lat_vecs_ddf.embedding_dim,
            )
        )
    for epoch in range(start_epoch, num_epochs + 1):
        start = time.time()

        loggerFunc("epoch {}...".format(epoch))

        decoder_ddf.train()
        adjust_learning_rate(lr_schedules, optimizer_ddf, epoch, inSchedule=lrSchedPoints)
        if not no_net_sdf:
            decoder_sdf.train()
            adjust_learning_rate(lr_schedules, optimizer_sdf, epoch)
        batch_count = 0
        startBatchTimer = time.time()
        for dsdf_data, indices in sdf_loader:
            if num_scenes == 1:
                indices = torch.tensor([indices])

            endBatchTimer = time.time()
            loggerFunc("Time taken for this mini batch: {}".format(endBatchTimer-startBatchTimer))
            startBatchTimer = time.time()

            dsdf_data = dsdf_data.float()
            dsdf_data = dsdf_data.reshape(-1, dsdf_data.shape[-1]).float()

            num_ddf_samples = dsdf_data.shape[0]
            loggerFunc("Batch Count: {} Training samples: {}".format(batch_count,num_ddf_samples))

            batch_count += 1
            dsdf_data.requires_grad = False

            dsdf_xyz = dsdf_data[:, 0:3]
            dsdf_dirs = dsdf_data[:, 3:6]
            dsdf_dist_gt = dsdf_data[:, 6].unsqueeze(1)
            dsdf_hitmiss_gt = dsdf_data[:, 7].unsqueeze(1)
            sdf_xyz = dsdf_data[:, 8:11]
            sdf_gt = dsdf_data[:, 11].unsqueeze(1)
            sdf_gt = torch.clamp(sdf_gt, -0.1, 0.1)
            sdf_xyz = torch.chunk(sdf_xyz, batch_split)
            sdf_gt = torch.chunk(sdf_gt, batch_split)

            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            dsdf_xyz = torch.chunk(dsdf_xyz, batch_split)
            dsdf_dirs = torch.chunk(dsdf_dirs, batch_split)
            dsdf_dist_gt = torch.chunk(dsdf_dist_gt, batch_split)
            dsdf_hitmiss_gt = torch.chunk(dsdf_hitmiss_gt, batch_split)

            batch_loss_ddf = 0.0
            batch_loss_sdf = 0.0


            optimizer_ddf.zero_grad()
            if not no_net_sdf:
                optimizer_sdf.zero_grad()
            for i in range(batch_split):
                batch_vecs_geom = lat_vecs(indices[i])
                if lat_vecs_seperate_ddf:
                    batch_vecs_geom_ddf = lat_vecs_ddf(indices[i])
                else:
                    batch_vecs_geom_ddf = lat_vecs(indices[i])

                ###########################DSDF#############################################################################
                input_geom_ddf = toCuda(torch.cat([batch_vecs_geom_ddf,dsdf_xyz[i], dsdf_dirs[i]], dim=1))
                pred_ddf = decoder_ddf(input_geom_ddf,inputDataDims=6)
                if missingDirs:
                    pred_hitmiss = pred_ddf[1]
                    pred_ddf = pred_ddf[0]
                predDists = toCuda(pred_ddf.squeeze())
                ddf_loss_temp = loss_l1_None(predDists[dsdf_hitmiss_gt[i].squeeze()==1], toCuda(dsdf_dist_gt[i][dsdf_hitmiss_gt[i].squeeze()==1].squeeze()))
                ddf_loss = ddf_loss_wt*ddf_loss_temp.mean()
                loggerFunc("geom_loss_ddf = {}".format(ddf_loss_temp.mean().item()))                 
                if missingDirs:
                    ddf_hitmiss_loss = loss_bce(pred_hitmiss.squeeze(), toCuda(dsdf_hitmiss_gt[i].squeeze()))
                    loggerFunc("hitmiss_loss_ddf = {}".format(ddf_hitmiss_loss.item()))
                    ddf_loss += ddf_loss_wt*ddf_hitmiss_loss

                if sdf_ddf_const_train:
                    predPC = toCuda(dsdf_xyz[i]) + toCuda(dsdf_dirs[i]) * predDists.unsqueeze(-1)
                    input_sdf_reg = toCuda(torch.cat([toCuda(batch_vecs_geom), predPC], dim=1))

                    if detach_ddf_sdf:
                        decoder_sdf_cloned.load_state_dict(decoder_sdf.state_dict())
                        pred_sdf = decoder_sdf_cloned(input_sdf_reg)
                    else:
                        pred_sdf = decoder_sdf(input_sdf_reg)
                    pred_sdf = torch.clamp(pred_sdf,-0.1,0.1)
                    sdf_ddf_loss = sdf_ddf_const_wt*torch.abs(pred_sdf).mean()
                    loggerFunc("sdf_ddf_loss = {}".format(sdf_ddf_loss.item()))                    
                    ddf_loss += sdf_ddf_loss                
                if tvlregfeats and (tvlregweightddf>0.0):
                    ddf_loss += tvlregweightddf*decoder_ddf.module.getFeatTVL()
                chunk_loss_ddf = ddf_loss
                ###########################SDF#############################################################################
                
                if not no_net_sdf:
                    input_geom_sdf = toCuda(torch.cat([batch_vecs_geom, sdf_xyz[i]], dim=1))
                    pred_sdf = decoder_sdf(input_geom_sdf)
                    pred_sdf = toCuda(pred_sdf.squeeze())
                    chunk_loss_sdf = 0


                    pred_sdf = torch.clamp(pred_sdf,-0.1,0.1)
                    sdf_loss = sdfLossWeight*torch.sum(loss_l1_None(pred_sdf, toCuda(sdf_gt[i].squeeze()))) / num_ddf_samples
                    chunk_loss_sdf += sdf_loss
                    loggerFunc("geom_loss_sdf = {}".format(sdf_loss.item()))
                    if tvlregfeats and tvlregweight>0:
                        chunk_loss_sdf += tvlregweight*decoder_sdf.module.getFeatTVL()
                else:
                    chunk_loss_sdf = torch.tensor(0).cuda()
                chunk_loss_vecs = 0
                if do_code_regularization:
                    l2_size_loss_geom = torch.sum(torch.norm(batch_vecs_geom, dim=1))
                    reg_loss_geom = (
                                            code_reg_lambda * min(1, epoch / 100) * l2_size_loss_geom
                                    ) / num_ddf_samples
                    chunk_loss_vecs = toCuda(reg_loss_geom)

                    if lat_vecs_seperate_ddf:
                        l2_size_loss_geom = torch.sum(torch.norm(batch_vecs_geom_ddf, dim=1))
                        reg_loss_geom_ddf = (
                                                code_reg_lambda * min(1, epoch / 100) * l2_size_loss_geom
                                        ) / num_ddf_samples
                        chunk_loss_vecs += toCuda(reg_loss_geom_ddf)

                # geometry backpropagation
                totalLoss = chunk_loss_sdf+chunk_loss_vecs+chunk_loss_ddf
                torch.autograd.set_detect_anomaly(True)
                totalLoss.backward()
                batch_loss_ddf += chunk_loss_ddf.item()
                batch_loss_sdf += chunk_loss_sdf.item()
                loggerFunc("Batch Loss DDF = {}".format(batch_loss_ddf))
                loggerFunc("Batch Loss SDF = {}".format(batch_loss_sdf))
                loss_log.append(batch_loss_sdf+batch_loss_ddf)
                sys.stdout.flush()
            if not no_net_sdf:
                optimizer_sdf.step()
            optimizer_ddf.step()
        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)
        loggerFunc("Time taken for this epoch: {}".format(seconds_elapsed))

        if epoch in checkpoints:
            save_checkpoints(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                timing_log,
                epoch,
            )

        if (epoch % log_frequency == 0):
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                timing_log,
                epoch,
            )

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train the DDF (and SDF) model")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )
    arg_parser.add_argument(
        "--distributedTrain",
        "-d",
        dest="distributedtrain",
        default=False,
        help="Distributed training",
    )

    DDF.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    DDF.configure_logging(args)
    distributedtrain = args.distributedtrain
    start = time.time()
    sdf_dataset=getDataset(args.experiment_directory)
    world_size = torch.cuda.device_count()
    if args.continue_from == "newest":
        print("Getting latest or the highest iteration model")
        tempfiles = os.listdir(ws.get_model_params_dir(args.experiment_directory, True))
        if len(tempfiles)>1:
            tempfidx = len("_ddf.pth")
            tempfiles = [x[:-tempfidx] for x in tempfiles if x.endswith("_ddf.pth")]
            tempfiles = sorted(tempfiles,reverse=True)
            newestCheckpoint = tempfiles[0]
            args.continue_from = newestCheckpoint
            print("Continuing from "+ args.continue_from)
        else:
            args.continue_from = None
            print("Staring from epoch 0")
    if (world_size>1) and distributedtrain:
        #get a free socket
        freePort = random.randint(10000,60000)
        print("Using port " + str(freePort))

        torch.multiprocessing.spawn(main_function,
            args=(distributedtrain, world_size,freePort,sdf_dataset,args.experiment_directory, args.continue_from, int(args.batch_split)),
            nprocs=world_size,
            join=True)
    else:
        main_function(0,distributedtrain,1,None,sdf_dataset,args.experiment_directory, args.continue_from, int(args.batch_split))
    end = time.time()
    print("Total time taken: {}".format(end-start))
