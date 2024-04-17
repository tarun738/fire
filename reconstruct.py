import torch
import DDF
import DDF.workspace as ws
import logging
from train_dsdf import ExpLearningRateSchedule
from networks.sdfnet import FeaturePlanes
def reconstructLatent(
  decoder_points,
  decoder_dirs,
  num_iterations,
  latent_size,
  test_sdf,
  stat,
  clamp_dist,
  num_samples=16384,
  lr=5e-3,
  l2reg=True,
  learnFeats =False,
  reconDDF = True,
 ):
    learnDDFFeats = learnFeats
    globalFeats = False
    # flrSDF = 1e-3
    # flrDDF = 1e-3
    flrSDF = 1e-5
    flrDDF = 1e-5
    num_iterations = 1000
    learningRateScheduler = ExpLearningRateSchedule(lr, num_iterations,0.1)
    def adjust_learning_rate( optimizer, iteration ):
        lr = learningRateScheduler.get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
    latent.requires_grad = True
    optvars = [
                {
                    "params": latent,
                    "lr": lr,
                }]

    if (not decoder_dirs.fdim ==0) and learnDDFFeats:
        addFeats = FeaturePlanes(decoder_dirs.fdim, decoder_dirs.fsize, decoder_dirs.input_dim, tvlsqrt=decoder_dirs.tvlsqrt, globalFeats=globalFeats,init_mul=0.0).cuda()
        for i in range(len(addFeats.fm)):
            addFeats.fm[i].data = decoder_dirs.features_planes.fm[i].data.clone()
        # addFeats.
        addFeats.requires_grad = True
        optvars.append(
                {
                    "params": addFeats.parameters(),
                    "lr": flrSDF,
                })
    else:
        addFeats = None
    if (not decoder_points.fdim ==0) and learnDDFFeats and reconDDF:
        addFeatsDDF = FeaturePlanes(decoder_points.fdim, decoder_points.fsize, decoder_points.input_dim, tvlsqrt=decoder_points.tvlsqrt, globalFeats=globalFeats,init_mul=0.0).cuda()
        for i in range(len(addFeatsDDF.fm)):
            addFeatsDDF.fm[i].data  = decoder_points.features_planes.fm[i].data.clone()
        addFeatsDDF.requires_grad = True
        optvars.append(
                {
                    "params": addFeatsDDF.parameters(),
                    "lr": flrDDF,
                })
    else:
        addFeatsDDF = None


    optimizer = torch.optim.Adam(optvars)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss() # gives the mean of L1 differences
    loss_bce = torch.nn.BCELoss() # gives the mean of L1 differences
    # import pdb; pdb.set_trace()
    for e in range(num_iterations):

        decoder_points.eval()
        decoder_dirs.eval()
        dsdf_data = DDF.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples, missingDirs=True).cuda()
        xyzdirs = dsdf_data[:, 0:6]
        dsdf_gt = dsdf_data[:, 6]
        dsdf_hitmiss = dsdf_data[:, 7]
        xyz_sdf = dsdf_data[:,8:11]
        sdf_gt = dsdf_data[:,11]
        color_gt = dsdf_data[:,11:]
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        # adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)
        # if e > 100:
        #     optimizer.param_groups[0]['lr'] = 0.0
        optimizer.zero_grad()
        loss = 0

        # latent = latent.expand(num_samples, -1)
        if reconDDF:
            inputsDDF = torch.cat([latent.repeat(xyzdirs.shape[0],1), xyzdirs], dim=1).cuda()
            pred_dsdf = decoder_points(inputsDDF,inputDataDims=6,getColor=False, addFeats=addFeatsDDF, inglobfeats=globalFeats)
            if decoder_points.output_dim > 1:
                pred_hitmiss = pred_dsdf[1]
                pred_dsdf = pred_dsdf[0]
            else:
                pred_hitmiss = None
            ddfLoss = loss_l1(pred_dsdf[dsdf_hitmiss==1], dsdf_gt[dsdf_hitmiss==1])
            loss += ddfLoss
            ddfLossLog = ddfLoss.item()
            if pred_hitmiss is not None:
                ddfHitLoss = loss_bce(pred_hitmiss.squeeze().cuda(),dsdf_hitmiss.squeeze().cuda())
                loss += ddfHitLoss
                ddfHitLossLog = ddfHitLoss.item()
        else:
            ddfHitLossLog = 0.0
            ddfLossLog = 0.0

        inputsSDF = torch.cat([latent.repeat(xyz_sdf.shape[0],1), xyz_sdf], dim=1).cuda()
        if decoder_dirs.hasColor:
            pred_sdf, pred_color = decoder_dirs(inputsSDF,inputDataDims=3)
        else:
            pred_sdf = decoder_dirs(inputsSDF,inputDataDims=3,addFeats=addFeats, inglobfeats=globalFeats)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        sdfLoss = loss_l1(pred_sdf, sdf_gt)
        loss += sdfLoss
        if decoder_dirs.hasColor:
            pred_color = torch.clamp(pred_color, 0.0, 1.0)
            loss += 0.5*loss_l1(pred_color,color_gt)
        if l2reg:
            latNorm = torch.mean(latent.pow(2))
            loss += 1e-4 * latNorm
            if addFeats is not None:
                loss += 1e2 * addFeats.getTVL()
            if addFeatsDDF is not None:
                loss += 1e2 * addFeatsDDF.getTVL()
                # loss += 1e0 * torch.abs(addFeats.fm[0]-decoder_dirs.features_planes.fm[0].data).mean()
                # loss += 1e0 * torch.abs(addFeats.fm[1]-decoder_dirs.features_planes.fm[1].data).mean()
                # loss += 1e0 * torch.abs(addFeats.fm[2]-decoder_dirs.features_planes.fm[2].data).mean()
        loss.backward()
        optimizer.step()

        if (e % 50 == 0) or (e == num_iterations-1):
            logging.info("Epoch {}, SDFLoss: {:4f}, DDFLoss: {:4f}, ddfHitLoss: {:4f}, Loss: {:4f}, Latent Norm: {:4f} ".format(e,sdfLoss.item(), ddfLossLog,ddfHitLossLog,loss.item(),latNorm))
        loss_num = loss.cpu().data.numpy()

    return loss_num, [latent.detach(),[addFeats,addFeatsDDF]]
